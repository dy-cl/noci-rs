// nonorthogonalwicks/gpu/types.rs
//! GPU metadata for compact nonorthogonal Wick storage.

// Crate-root imports.
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
