// nonorthogonalwicks/gpu/storage.rs
//! GPU-owned compact nonorthogonal Wick storage.

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext};
use crate::noci::NOCIScalar;
use crate::nonorthogonalwicks::WicksRequirements;

// Parent/sibling imports.
use super::types::{DeviceWicksShared, PairMeta, PairOffset};
use super::view::WicksView;

/// GPU owner for compact Wick data required by factorised NOCI-PT2 one-body evaluation.
pub(crate) struct WicksShared<T: NOCIScalar> {
    /// Requested Wick capability set.
    requirements: WicksRequirements,
    /// Compact same-spin tensor slab.
    slab: Vec<T>,
    /// Number of reference determinants.
    nref: usize,
    /// Per-reference-pair offsets into the compact slab.
    off: Vec<PairOffset>,
    /// Per-reference-pair scalar metadata.
    meta: Vec<PairMeta<T>>,
}

impl<T: NOCIScalar> WicksShared<T> {
    /// Construct compact GPU Wick storage from packed host buffers.
    /// # Arguments:
    /// - `requirements`: Required Wick capability set.
    /// - `slab`: Compact same-spin tensor slab.
    /// - `nref`: Number of reference determinants.
    /// - `off`: Per-reference-pair offsets into `slab`.
    /// - `meta`: Per-reference-pair scalar metadata.
    /// # Returns
    /// - `WicksShared<T>`: GPU Wick storage owner.
    pub(crate) fn new(
        requirements: WicksRequirements,
        slab: Vec<T>,
        nref: usize,
        off: Vec<PairOffset>,
        meta: Vec<PairMeta<T>>,
    ) -> Self {
        Self {
            requirements,
            slab,
            nref,
            off,
            meta,
        }
    }

    /// Borrow the GPU Wick view.
    /// # Arguments:
    /// - `self`: GPU Wick storage owner.
    /// # Returns
    /// - `WicksView<'_, T>`: Read-only compact Wick view.
    pub(crate) fn view(&self) -> WicksView<'_, T> {
        WicksView {
            requirements: self.requirements,
            slab: &self.slab,
            nref: self.nref,
            off: &self.off,
            meta: &self.meta,
        }
    }

    /// Upload the real-valued NOCI-PT2 Wick storage to CubeCL device buffers.
    /// The device storage contains only `X`, `Y`, `ff`, `f0f`, `phase`, `tilde_s_prod`, `m`,
    /// `nmo` and `nocc` for the alpha-alpha and beta-beta same-spin sectors.
    /// # Arguments:
    /// - `self`: Host-packed Wick storage.
    /// - `context`: CubeCL context owning the target device.
    /// # Returns
    /// - `DeviceWicksShared`: Device-resident compact Wick buffers.
    pub(crate) fn upload_f64(
        &self,
        context: &GpuContext,
    ) -> DeviceWicksShared
    where
        T: Into<f64> + Copy,
    {
        let slab = self
            .slab
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<_>>();
        let mut x_off = Vec::with_capacity(self.off.len() * 4);
        let mut y_off = Vec::with_capacity(self.off.len() * 4);
        let mut ff_off = Vec::with_capacity(self.off.len() * 8);
        let mut phase = Vec::with_capacity(self.meta.len() * 2);
        let mut tilde_s_prod = Vec::with_capacity(self.meta.len() * 2);
        let mut f0f = Vec::with_capacity(self.meta.len() * 4);
        let mut m = Vec::with_capacity(self.meta.len() * 2);
        let mut nmo = Vec::with_capacity(self.meta.len() * 2);
        let mut nocc = Vec::with_capacity(self.meta.len() * 2);

        for (off, meta) in self.off.iter().zip(&self.meta) {
            push_same_offsets(&mut x_off, &mut y_off, &mut ff_off, &off.aa);
            push_same_meta(
                &mut phase,
                &mut tilde_s_prod,
                &mut f0f,
                &mut m,
                &mut nmo,
                &mut nocc,
                &meta.aa,
            );
            push_same_offsets(&mut x_off, &mut y_off, &mut ff_off, &off.bb);
            push_same_meta(
                &mut phase,
                &mut tilde_s_prod,
                &mut f0f,
                &mut m,
                &mut nmo,
                &mut nocc,
                &meta.bb,
            );
        }

        DeviceWicksShared {
            slab: GpuBuffer::from_slice(context, &slab),
            x_off: GpuBuffer::from_slice(context, &x_off),
            y_off: GpuBuffer::from_slice(context, &y_off),
            ff_off: GpuBuffer::from_slice(context, &ff_off),
            phase: GpuBuffer::from_slice(context, &phase),
            tilde_s_prod: GpuBuffer::from_slice(context, &tilde_s_prod),
            f0f: GpuBuffer::from_slice(context, &f0f),
            m: GpuBuffer::from_slice(context, &m),
            nmo: GpuBuffer::from_slice(context, &nmo),
            nocc: GpuBuffer::from_slice(context, &nocc),
        }
    }
}

/// Append primitive offsets for one same-spin sector.
/// # Arguments:
/// - `x_off`: Flattened X offsets.
/// - `y_off`: Flattened Y offsets.
/// - `ff_off`: Flattened transposed Fock offsets.
/// - `same`: Host same-spin offsets.
/// # Returns
/// - `()`: Appends offsets in kernel indexing order.
fn push_same_offsets(
    x_off: &mut Vec<u32>,
    y_off: &mut Vec<u32>,
    ff_off: &mut Vec<u32>,
    same: &super::types::SameSpinOffset,
) {
    x_off.extend(same.x.map(checked_u32));
    y_off.extend(same.y.map(checked_u32));
    for mi in 0..2 {
        for mj in 0..2 {
            ff_off.push(checked_u32(same.ff[mi][mj]));
        }
    }
}

/// Append primitive metadata for one same-spin sector.
/// # Arguments:
/// - `phase`: Flattened orbital-pairing phases.
/// - `tilde_s_prod`: Flattened non-zero singular-value products.
/// - `f0f`: Flattened scalar current-Fock intermediates.
/// - `m`: Flattened zero-overlap counts.
/// - `nmo`: Flattened orbital dimensions.
/// - `nocc`: Flattened occupied dimensions.
/// - `same`: Host same-spin metadata.
/// # Returns
/// - `()`: Appends metadata in kernel indexing order.
fn push_same_meta<T: NOCIScalar + Into<f64> + Copy>(
    phase: &mut Vec<f64>,
    tilde_s_prod: &mut Vec<f64>,
    f0f: &mut Vec<f64>,
    m: &mut Vec<u32>,
    nmo: &mut Vec<u32>,
    nocc: &mut Vec<u32>,
    same: &super::types::SameSpinMeta<T>,
) {
    phase.push(same.phase.into());
    tilde_s_prod.push(same.tilde_s_prod);
    f0f.extend(same.f0f.map(Into::into));
    m.push(checked_u32(same.m));
    nmo.push(checked_u32(same.nmo));
    nocc.push(checked_u32(same.nocc));
}

/// Convert `usize` metadata to `u32`, failing before any silent truncation.
/// # Arguments:
/// - `value`: Host metadata value.
/// # Returns
/// - `u32`: Checked device-width value.
fn checked_u32(value: usize) -> u32 {
    u32::try_from(value).expect("GPU Wick metadata exceeds u32 device representation")
}
