// nonorthogonalwicks/gpu/storage.rs
//! GPU-owned compact nonorthogonal Wick storage.

// Crate-root imports.
use crate::noci::NOCIScalar;
use crate::nonorthogonalwicks::WicksRequirements;

// Parent/sibling imports.
use super::types::{PairMeta, PairOffset};
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
}
