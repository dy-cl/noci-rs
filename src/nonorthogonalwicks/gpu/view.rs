// nonorthogonalwicks/gpu/view.rs
//! GPU-compatible views over compact nonorthogonal Wick storage.

// Standard library imports.
use std::ops::Deref;

// Crate-root imports.
use crate::noci::NOCIScalar;
use crate::nonorthogonalwicks::WicksRequirements;

// Parent/sibling imports.
use super::types::{PairMeta, PairOffset, SameSpinMeta, SameSpinOffset};

/// Read-only compact GPU Wick view.
#[derive(Clone, Copy)]
pub(crate) struct WicksView<'a, T: NOCIScalar> {
    /// Requested Wick capability set.
    pub(crate) requirements: WicksRequirements,
    /// Compact same-spin tensor slab.
    pub(crate) slab: &'a [T],
    /// Number of reference determinants.
    pub(crate) nref: usize,
    /// Per-reference-pair offsets into the compact slab.
    pub(crate) off: &'a [PairOffset],
    /// Per-reference-pair scalar metadata.
    pub(crate) meta: &'a [PairMeta<T>],
}

impl<'a, T: NOCIScalar> WicksView<'a, T> {
    /// Map ordered reference-pair indices `(x,w)` to a flattened pair index.
    /// # Arguments:
    /// - `self`: GPU Wick view.
    /// - `lp`: Bra-reference index `x`.
    /// - `gp`: Ket-reference index `w`.
    /// # Returns
    /// - `usize`: Flattened ordered pair index.
    fn idx(
        &self,
        lp: usize,
        gp: usize,
    ) -> usize {
        lp * self.nref + gp
    }

    /// Return grouped alpha-alpha and beta-beta same-spin views for an ordered reference pair.
    /// # Arguments:
    /// - `self`: GPU Wick view.
    /// - `lp`: Bra-reference index `x`.
    /// - `gp`: Ket-reference index `w`.
    /// # Returns
    /// - `WicksPairView<'_, T>`: Same-spin views for the ordered pair.
    pub(crate) fn pair(
        &self,
        lp: usize,
        gp: usize,
    ) -> WicksPairView<'_, T> {
        let idx = self.idx(lp, gp);
        let meta = &self.meta[idx];
        let off = &self.off[idx];
        WicksPairView {
            aa: SameSpinView {
                meta: &meta.aa,
                slab: self.slab,
                off: &off.aa,
            },
            bb: SameSpinView {
                meta: &meta.bb,
                slab: self.slab,
                off: &off.bb,
            },
        }
    }
}

/// GPU same-spin Wick view.
#[derive(Clone, Copy)]
pub(crate) struct SameSpinView<'a, T: NOCIScalar> {
    /// Same-spin metadata.
    pub(crate) meta: &'a SameSpinMeta<T>,
    /// Compact tensor slab.
    pub(crate) slab: &'a [T],
    /// Same-spin offsets.
    pub(crate) off: &'a SameSpinOffset,
}

impl<T: NOCIScalar> Deref for SameSpinView<'_, T> {
    type Target = SameSpinMeta<T>;

    /// Borrow same-spin metadata for transparent field access.
    /// # Arguments:
    /// - `self`: Same-spin GPU Wick view.
    /// # Returns
    /// - `&SameSpinMeta<T>`: Same-spin metadata.
    fn deref(&self) -> &Self::Target {
        self.meta
    }
}

impl<T: NOCIScalar> SameSpinView<'_, T> {
    /// Return the molecular-orbital dimension.
    /// # Arguments:
    /// - `self`: Same-spin GPU Wick view.
    /// # Returns
    /// - `usize`: Matrix dimension `nmo`.
    pub(crate) fn n(&self) -> usize {
        self.nmo
    }

    /// Return `X^{(m_i)}` as a flat row-major slice.
    /// # Arguments:
    /// - `self`: Same-spin GPU Wick view.
    /// - `mi`: Fundamental-contraction assignment.
    /// # Returns
    /// - `&[T]`: Flat row-major `X^{(m_i)}` matrix.
    pub(crate) fn x_slice(
        &self,
        mi: usize,
    ) -> &[T] {
        let off = self.off.x[mi];
        &self.slab[off..off + self.n() * self.n()]
    }

    /// Return `Y^{(m_i)}` as a flat row-major slice.
    /// # Arguments:
    /// - `self`: Same-spin GPU Wick view.
    /// - `mi`: Fundamental-contraction assignment.
    /// # Returns
    /// - `&[T]`: Flat row-major `Y^{(m_i)}` matrix.
    pub(crate) fn y_slice(
        &self,
        mi: usize,
    ) -> &[T] {
        let off = self.off.y[mi];
        &self.slab[off..off + self.n() * self.n()]
    }

    /// Return transposed current-Fock one-column intermediate as `[z,r]` flat storage.
    /// The `[z,r]` ordering preserves the CPU `ff_t_slice` convention so replacement columns
    /// with fixed `z` remain contiguous.
    /// # Arguments:
    /// - `self`: Same-spin GPU Wick view.
    /// - `mi`: First fundamental-contraction assignment.
    /// - `mj`: Second fundamental-contraction assignment.
    /// # Returns
    /// - `&[T]`: Flat transposed `\mathcal F^{(m_i,m_j)}` matrix.
    pub(crate) fn ff_t_slice(
        &self,
        mi: usize,
        mj: usize,
    ) -> &[T] {
        let off = self.off.ff[mi][mj];
        &self.slab[off..off + self.n() * self.n()]
    }
}

/// GPU pair view containing the same-spin sectors required by NOCI-PT2.
#[derive(Clone, Copy)]
pub(crate) struct WicksPairView<'a, T: NOCIScalar> {
    /// Alpha-alpha same-spin view.
    pub(crate) aa: SameSpinView<'a, T>,
    /// Beta-beta same-spin view.
    pub(crate) bb: SameSpinView<'a, T>,
}
