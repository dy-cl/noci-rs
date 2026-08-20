// nonorthogonalwicks/types.rs
// External crate imports.
use serde::{Deserialize, Serialize};

// Crate-root imports.
use crate::noci::NOCIScalar;

/// Numbers of zero-overlap occupied-orbital pairs used to allocate only the pair-adaptive
/// rank-four intermediates that can contribute for an ordered reference pair (x,w).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PairZeroCounts {
    /// `Number m_\alpha of alpha-spin zero-overlap occupied-orbital pairs.`
    pub(crate) ma: usize,
    /// `Number m_\beta of beta-spin zero-overlap occupied-orbital pairs.`
    pub(crate) mb: usize,
}

/// Metadata and scalar same-spin intermediates stored outside the shared tensor slab for one
/// spin sector of an ordered reference pair (x,w).
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "T: NOCIScalar")]
pub(crate) struct SameSpinMeta<T: NOCIScalar> {
    /// Product of the non-zero occupied-orbital singular values,
    /// `\prod_{\{i\mid{}^{xw}\tilde S_{\sigma i}\neq0\}}{}^{xw}\tilde S_{\sigma i}.`
    pub(crate) tilde_s_prod: f64,
    /// `Orbital-pairing phase \phi_\sigma^{xw}; together with tilde_s_prod this gives the`
    /// `spin-sector reduced overlap {}^{xw}\tilde S_\sigma.`
    pub(crate) phase: T,
    /// `Number m_\sigma of zero-overlap occupied-orbital pairs in this spin sector.`
    pub(crate) m: usize,
    /// Number of molecular orbitals in one reference orbital set.
    pub(crate) nmo: usize,
    /// Number of occupied orbitals in this spin sector.
    pub(crate) nocc: usize,
    /// `Scalar one-body intermediates {}^xF_0^{(m_1)} constructed from the current`
    /// generalised-Fock operator.
    pub(crate) f0f: [T; 2],
    /// `Scalar one-body intermediates {}^xF_0^{(m_1)} constructed from the one-electron`
    /// Hamiltonian.
    pub(crate) f0h: [T; 2],
    /// `Scalar same-spin two-body intermediates {}^xV_0^{(m_1,m_2)}, stored by`
    /// `m_1 + m_2. The middle entry contains the combined (0,1) and (1,0) contribution.`
    pub(crate) v0: [T; 3],
}

impl<T: NOCIScalar> Default for SameSpinMeta<T> {
    fn default() -> Self {
        Self {
            tilde_s_prod: 0.0,
            phase: <T as From<f64>>::from(0.0),
            m: 0,
            nmo: 0,
            nocc: 0,
            f0f: [<T as From<f64>>::from(0.0); 2],
            f0h: [<T as From<f64>>::from(0.0); 2],
            v0: [<T as From<f64>>::from(0.0); 3],
        }
    }
}

/// Metadata and scalar different-spin intermediates stored outside the shared tensor slab for
/// an ordered reference pair (x,w).
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "T: NOCIScalar")]
pub(crate) struct DiffSpinMeta<T: NOCIScalar> {
    /// Number of molecular orbitals in one reference orbital set.
    pub(crate) nmo: usize,
    /// Scalar different-spin intermediates
    /// `{}^xV_{\alpha\beta,0}^{(m_{\alpha0},m_{\beta0})}.`
    pub(crate) vab0: [[T; 2]; 2],
    /// Scalar different-spin intermediates
    /// `{}^xV_{\beta\alpha,0}^{(m_{\beta0},m_{\alpha0})}.`
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

/// Offsets for the same-spin intermediates belonging to one spin sector of an ordered reference
/// pair (x,w) in the shared contiguous tensor slab.
#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
pub(crate) struct SameSpinOffset {
    /// `Offsets to the X^{(m_i)} fundamental-contraction matrices.`
    pub(in crate::nonorthogonalwicks) x: [usize; 2],
    /// `Offsets to the Y^{(m_i)} fundamental-contraction matrices.`
    pub(in crate::nonorthogonalwicks) y: [usize; 2],
    /// `Offsets to X^{(m_i)} represented in the external RDM basis.`
    pub(in crate::nonorthogonalwicks) xrdm: [usize; 2],
    /// `Offsets to Y^{(m_i)} represented in the external RDM basis.`
    pub(in crate::nonorthogonalwicks) yrdm: [usize; 2],
    /// `Offsets to the transposed one-column intermediates \mathcal F^{(m_i,m_j)}`
    /// constructed from the one-electron Hamiltonian.
    pub(in crate::nonorthogonalwicks) fh: [[usize; 2]; 2],
    /// `Offsets to the transposed one-column intermediates \mathcal F^{(m_i,m_j)}`
    /// constructed from the current generalised-Fock operator.
    pub(crate) ff: [[usize; 2]; 2],
    /// Offsets to the transposed same-spin one-column intermediates
    /// `\mathcal V^{(m_1,m_2,m_3)}, stored as v[m_1][m_3][m_2].`
    pub(in crate::nonorthogonalwicks) v: [[[usize; 2]; 2]; 2],
    /// Offset to the transposed precombined `m_\alpha = m_\beta = 0` Hamiltonian
    /// one-column intermediate for this spin sector.
    pub(in crate::nonorthogonalwicks) hcol0: usize,
    /// Offsets to the ten symmetry-unique same-spin
    /// `\mathcal J^{(m_1,m_2,m_3,m_4)} tensors stored in evaluator axis order.`
    pub(in crate::nonorthogonalwicks) j: [usize; 10],
}

/// Offsets for the different-spin intermediates of an ordered reference pair (x,w) in the
/// shared contiguous tensor slab.
#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
pub(crate) struct DiffSpinOffset {
    /// Offsets to the transposed alpha-spin one-column intermediates
    /// `\mathcal V^\alpha{}^{(m_{\alpha0},m_{\beta0},m_{\alpha z})}.`
    pub(in crate::nonorthogonalwicks) vab: [[[usize; 2]; 2]; 2],
    /// Offsets to the transposed beta-spin one-column intermediates
    /// `\mathcal V^\beta{}^{(m_{\beta0},m_{\alpha0},m_{\beta y})}.`
    pub(in crate::nonorthogonalwicks) vba: [[[usize; 2]; 2]; 2],
    /// Offsets to the different-spin two-column intermediates
    /// `\mathcal{II}^{(m_{\alpha0},m_{\alpha z},m_{\beta0},m_{\beta y})}.`
    pub(in crate::nonorthogonalwicks) iiab: [[[[usize; 2]; 2]; 2]; 2],
}

/// Offset tables for the alpha-alpha, beta-beta and alpha-beta intermediates of one ordered
/// reference pair (x,w).
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub(crate) struct PairOffset {
    /// Same-spin alpha-alpha offset table.
    pub(crate) aa: SameSpinOffset,
    /// Same-spin beta-beta offset table.
    pub(crate) bb: SameSpinOffset,
    /// Different-spin alpha-beta offset table.
    pub(crate) ab: DiffSpinOffset,
}

/// Scalar metadata for the alpha-alpha, beta-beta and alpha-beta intermediates of one ordered
/// reference pair (x,w).
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
