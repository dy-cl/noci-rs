// nonorthogonalwicks/gpu/types.rs
//! GPU metadata for compact nonorthogonal Wick storage.

// Crate-root imports.
use crate::gpu::GpuBuffer;
use crate::noci::NOCIScalar;

/// GPU offsets for one same-spin sector in the compact NOCI-PT2 Wick slab.
#[derive(Clone, Copy, Default)]
pub(crate) struct SameSpinOffset {
    /// Offsets to `X^{(m_i)}` matrices.
    pub(crate) x: [usize; 2],
    /// Offsets to `Y^{(m_i)}` matrices.
    pub(crate) y: [usize; 2],
    /// Offsets to transposed current-Fock one-column intermediates.
    pub(crate) ff: [[usize; 2]; 2],
}

/// GPU metadata for one same-spin sector in the compact NOCI-PT2 Wick slab.
#[derive(Clone, Copy)]
pub(crate) struct SameSpinMeta<T: NOCIScalar> {
    /// Product of non-zero occupied-orbital singular values.
    pub(crate) tilde_s_prod: f64,
    /// Orbital-pairing phase.
    pub(crate) phase: T,
    /// Number of zero-overlap occupied-orbital pairs.
    pub(crate) m: usize,
    /// Number of molecular orbitals.
    pub(crate) nmo: usize,
    /// Number of occupied orbitals.
    pub(crate) nocc: usize,
    /// Scalar current-Fock one-body intermediates.
    pub(crate) f0f: [T; 2],
}

/// GPU offsets for one ordered reference pair.
#[derive(Clone, Copy, Default)]
pub(crate) struct PairOffset {
    /// Alpha-alpha same-spin offsets.
    pub(crate) aa: SameSpinOffset,
    /// Beta-beta same-spin offsets.
    pub(crate) bb: SameSpinOffset,
}

/// GPU metadata for one ordered reference pair.
#[derive(Clone, Copy)]
pub(crate) struct PairMeta<T: NOCIScalar> {
    /// Alpha-alpha same-spin metadata.
    pub(crate) aa: SameSpinMeta<T>,
    /// Beta-beta same-spin metadata.
    pub(crate) bb: SameSpinMeta<T>,
}

/// Device-side primitive Wick buffers for real NOCI-PT2 same-spin evaluation.
pub(crate) struct DeviceWicksShared {
    /// Compact same-spin tensor slab containing only `X`, `Y` and transposed current-Fock `ff`.
    pub(crate) slab: GpuBuffer<f64>,
    /// Flattened offsets for `X^(0)` and `X^(1)` per ordered pair and spin.
    pub(crate) x_off: GpuBuffer<u32>,
    /// Flattened offsets for `Y^(0)` and `Y^(1)` per ordered pair and spin.
    pub(crate) y_off: GpuBuffer<u32>,
    /// Flattened offsets for `ff^(0,0)`, `ff^(0,1)`, `ff^(1,0)` and `ff^(1,1)`.
    pub(crate) ff_off: GpuBuffer<u32>,
    /// Orbital-pairing phase per ordered pair and spin.
    pub(crate) phase: GpuBuffer<f64>,
    /// Product of non-zero occupied-orbital singular values per ordered pair and spin.
    pub(crate) tilde_s_prod: GpuBuffer<f64>,
    /// Scalar current-Fock one-body intermediates `f0f[0]` and `f0f[1]`.
    pub(crate) f0f: GpuBuffer<f64>,
    /// Number of zero-overlap occupied-orbital pairs per ordered pair and spin.
    pub(crate) m: GpuBuffer<u32>,
    /// Number of molecular orbitals per ordered pair and spin.
    pub(crate) nmo: GpuBuffer<u32>,
    /// Number of occupied orbitals per ordered pair and spin.
    pub(crate) nocc: GpuBuffer<u32>,
}
