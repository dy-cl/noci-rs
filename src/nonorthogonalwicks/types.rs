// nonorthogonalwicks/types.rs
use serde::{Deserialize, Serialize};

use crate::noci::NOCIScalar;

/// Storage for same-spin metadata and lightweight scalars that we can store outside the shared
/// memory region.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "T: NOCIScalar")]
pub(crate) struct SameSpinMeta<T: NOCIScalar> {
    /// Product of the non-zero singular values for this same-spin block.
    pub(crate) tilde_s_prod: f64,
    /// Overall phase associated with this same-spin block.
    pub(crate) phase: T,
    /// Number of zero-overlap orbital pairs in the biorthogonal basis.
    pub(crate) m: usize,
    /// Number of molecular orbitals for this same-spin block.
    pub(crate) nmo: usize,
    /// Zeroth-order Fock one-body scalar contributions for the two branch choices.
    pub(crate) f0f: [T; 2],
    /// Zeroth-order Hamiltonian one-body scalar contributions for the two branch choices.
    pub(crate) f0h: [T; 2],
    /// Zeroth-order two-body scalar contributions for the allowed branch combinations.
    pub(crate) v0: [T; 3],
}

impl<T: NOCIScalar> Default for SameSpinMeta<T> {
    fn default() -> Self {
        Self {
            tilde_s_prod: 0.0,
            phase: <T as From<f64>>::from(0.0),
            m: 0,
            nmo: 0,
            f0f: [<T as From<f64>>::from(0.0); 2],
            f0h: [<T as From<f64>>::from(0.0); 2],
            v0: [<T as From<f64>>::from(0.0); 3],
        }
    }
}

/// Storage for diff-spin metadata and lightweight scalars that we can store outside the shared
/// memory region.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "T: NOCIScalar")]
pub(crate) struct DiffSpinMeta<T: NOCIScalar> {
    /// Number of molecular orbitals for this different-spin block.
    pub(crate) nmo: usize,
    /// Zeroth-order mixed-spin Vab scalar contributions for the branch combinations.
    pub(crate) vab0: [[T; 2]; 2],
    /// Zeroth-order mixed-spin Vba scalar contributions for the branch combinations.
    pub(crate) vba0: [[T; 2]; 2],
}

impl<T: NOCIScalar> Default for DiffSpinMeta<T> {
    fn default() -> Self {
        Self {
            nmo: 0,
            vab0: [[<T as From<f64>>::from(0.0); 2]; 2],
            vba0: [[<T as From<f64>>::from(0.0); 2]; 2],
        }
    }
}

/// Storage for same-spin per reference-pair offset tables into the shared contiguous tensor storage.
#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
pub(crate) struct SameSpinOffset {
    /// Offsets to the X[mi] contraction matrices.
    pub(in crate::nonorthogonalwicks) x: [usize; 2],
    /// Offsets to the Y[mi] contraction matrices.
    pub(in crate::nonorthogonalwicks) y: [usize; 2],
    /// Offsets to the transposed Hamiltonian one-body F[mi][mj] intermediates.
    pub(in crate::nonorthogonalwicks) fh: [[usize; 2]; 2],
    /// Offsets to the transposed Fock one-body F[mi][mj] intermediates.
    pub(crate) ff: [[usize; 2]; 2],
    /// Offsets to the transposed same-spin V[mi][mj][mk] intermediates.
    pub(in crate::nonorthogonalwicks) v: [[[usize; 2]; 2]; 2],
    /// Offsets to the compressed same-spin J tensors.
    pub(in crate::nonorthogonalwicks) j: [usize; 10],
}

/// Storage for diff-spin per reference-pair offset tables into the shared contiguous tensor storage.
#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
pub(crate) struct DiffSpinOffset {
    /// Offsets to the transposed Vab[ma0][mb0][mak] intermediates.
    pub(in crate::nonorthogonalwicks) vab: [[[usize; 2]; 2]; 2],
    /// Offsets to the transposed Vba[mb0][ma0][mbk] intermediates.
    pub(in crate::nonorthogonalwicks) vba: [[[usize; 2]; 2]; 2],
    /// Offsets to the IIab[ma0][maj][mb0][mbj] tensors.
    pub(in crate::nonorthogonalwicks) iiab: [[[[usize; 2]; 2]; 2]; 2],
}

/// Storage for all per reference-pair pair offset tables.
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub(crate) struct PairOffset {
    /// Same-spin alpha-alpha offset table.
    pub(crate) aa: SameSpinOffset,
    /// Same-spin beta-beta offset table.
    pub(crate) bb: SameSpinOffset,
    /// Different-spin alpha-beta offset table.
    pub(crate) ab: DiffSpinOffset,
}

/// Storage for all pair metadata.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "T: NOCIScalar")]
pub(crate) struct PairMeta<T: NOCIScalar> {
    /// Same-spin alpha-alpha metadata.
    pub(crate) aa: SameSpinMeta<T>,
    /// Same-spin beta-beta metadata.
    pub(crate) bb: SameSpinMeta<T>,
    /// Different-spin alpha-beta metadata.
    pub(crate) ab: DiffSpinMeta<T>,
}

impl<T: NOCIScalar> Default for PairMeta<T> {
    fn default() -> Self {
        Self {
            aa: SameSpinMeta::default(),
            bb: SameSpinMeta::default(),
            ab: DiffSpinMeta::default(),
        }
    }
}
