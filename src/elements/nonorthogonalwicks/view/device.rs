// elements/nonorthogonalwicks/view/device.rs
//! `DeviceWicks`, the GPU counterpart of `WicksView` in `view.rs`: the intermediates of every ordered
//! pair of distinct references, for the GPU Wick kernels in `eval/device.cuh`. Each pair holds one
//! combination of intermediates per near-zero branch combination `(i_\alpha, i_\beta)`, each with
//! its own number `m_\sigma` of zero-overlap orbital pairs, exactly as `WicksView::combination`
//! returns them. Every stored matrix and tensor of a combination is copied into one device slab,
//! except the reduced density contractions, which no overlap, Fock or Hamiltonian element reads.
//! The generalised-Fock intermediates follow the current Fock, so a new `DeviceWicks` is built
//! whenever the Fock changes. Real scalars only.

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, DriverError};

// Crate-root imports.
use crate::NOCIScalar;

// Parent/sibling imports.
use super::WicksView;

/// Sum of the assignments `(m_1, m_2, m_{\text{other}}, m_{\text{fixed}})` of each stored
/// same-spin `\mathcal J` slot, in the order of `jslot`. A slot is stored when its sum is at most
/// `m`.
const JSUM: [usize; 10] = [0, 1, 1, 2, 2, 2, 3, 2, 3, 4];

/// Same-spin intermediates of one combination: offsets into the device slab, stored blocks only,
/// and scalars. Every field is eight bytes wide, matching the CUDA struct of the same name.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DeviceWicksSpin {
    /// Number `m` of zero-overlap orbital pairs.
    m: u64,
    /// Number of molecular orbitals.
    nmo: u64,
    /// Number of occupied orbitals.
    nocc: u64,
    /// Reduced reference overlap with its orbital-pairing phase, `{}^{xw}\tilde S`.
    pref: f64,
    /// `X^{(m_i)}`, row-major `nmo x nmo`.
    x: [u64; 2],
    /// `Y^{(m_i)}`, row-major `nmo x nmo`.
    y: [u64; 2],
    /// Generalised-Fock columns `\mathcal F^{(m_i,m_j)}`, stored transposed.
    ff: [[u64; 2]; 2],
    /// One-electron columns `\mathcal F^{h(m_i,m_j)}`, stored transposed.
    fh: [[u64; 2]; 2],
    /// Same-spin one-column `\mathcal V^{(m_i,m_j,m_k)}`, stored transposed.
    v: [[[u64; 2]; 2]; 2],
    /// Same-spin two-column `\mathcal J` slots, `nmo^4` entries each.
    j: [u64; 10],
    /// Combined one-column `\mathcal H`, stored transposed, only when both spins have `m = 0`.
    hcol0: u64,
    /// Generalised-Fock reference elements `F_0^{(m_1)}`.
    f0f: [f64; 2],
    /// One-electron reference elements `F_0^{h,(m_1)}`.
    f0h: [f64; 2],
    /// Same-spin two-electron reference elements `V_0^{(m_1+m_2)}`.
    v0: [f64; 3],
}

/// Mixed-spin intermediates of one combination, as offsets into the device slab and scalars.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DeviceWicksMixed {
    /// Orbital dimension of the mixed-spin intermediates.
    nab: u64,
    /// Alpha one-column `\mathcal V^{\alpha(m_{\alpha 0},m_{\beta 0},m_{\alpha z})}`, transposed.
    vab: [[[u64; 2]; 2]; 2],
    /// Beta one-column `\mathcal V^{\beta(m_{\beta 0},m_{\alpha 0},m_{\beta y})}`, transposed.
    vba: [[[u64; 2]; 2]; 2],
    /// Two-column `\mathcal{II}^{(m_{\alpha 0},m_{\alpha z},m_{\beta 0},m_{\beta y})}`, `nab^4`
    /// entries each, stored blocks only.
    ii: [[[[u64; 2]; 2]; 2]; 2],
    /// Reference elements `V_{\alpha\beta,0}^{(m_{\alpha 0},m_{\beta 0})}`.
    vab0: [[f64; 2]; 2],
    /// Reference elements `V_{\beta\alpha,0}^{(m_{\beta 0},m_{\alpha 0})}`.
    vba0: [[f64; 2]; 2],
}

/// Intermediates of one near-zero branch combination of a reference pair.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DeviceWicksCombination {
    /// Alpha and beta same-spin intermediates.
    spin: [DeviceWicksSpin; 2],
    /// Mixed-spin intermediates.
    mixed: DeviceWicksMixed,
}

/// Location of one ordered reference pair's combinations.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DeviceWicksPair {
    /// One when the GPU evaluates this pair, zero otherwise.
    evaluated: u64,
    /// Position of combination `(0, 0)`; combination `(i_\alpha, i_\beta)` is at
    /// `start + i_\alpha n_\beta + i_\beta`.
    start: u64,
    /// Number of alpha branches.
    na: u64,
    /// Number of beta branches.
    nb: u64,
}

// Plain data with a C layout, so the driver may copy them byte for byte.
unsafe impl DeviceRepr for DeviceWicksSpin {}
unsafe impl DeviceRepr for DeviceWicksMixed {}
unsafe impl DeviceRepr for DeviceWicksCombination {}
unsafe impl DeviceRepr for DeviceWicksPair {}

/// Wick intermediates of every pair of distinct references on a GPU.
pub(crate) struct DeviceWicks {
    /// Number of references.
    pub(crate) nref: usize,
    /// Whether the GPU evaluates each ordered reference pair, indexed as `x * nref + w`.
    pub(crate) parents: Vec<bool>,
    /// Location of each ordered reference pair's combinations.
    pub(crate) pairs: CudaSlice<DeviceWicksPair>,
    /// Intermediates of every combination.
    pub(crate) combinations: CudaSlice<DeviceWicksCombination>,
    /// Every copied block, located by the combinations.
    pub(crate) slab: CudaSlice<f64>,
}

impl DeviceWicks {
    /// Copy the intermediates of every pair of distinct references. Real scalars only.
    /// # Arguments:
    /// - `stream`: Stream of the GPU to copy to.
    /// - `view`: Wick intermediates with the current generalised Fock.
    /// # Returns:
    /// - `Result<DeviceWicks, DriverError>`: The intermediates on the GPU.
    pub(crate) fn new<T: NOCIScalar>(
        stream: &Arc<CudaStream>,
        view: &WicksView<T>,
    ) -> Result<Self, DriverError> {
        let nref = view.nref;
        let mut slab: Vec<f64> = Vec::new();
        let mut pairs = Vec::with_capacity(nref * nref);
        let mut combinations = Vec::new();
        let mut parents = Vec::with_capacity(nref * nref);

        // Append one block and return where it starts.
        let mut push = |values: &[T]| {
            let offset = slab.len() as u64;
            slab.extend(values.iter().map(|v| v.re()));
            offset
        };

        for x in 0..nref {
            for w in 0..nref {
                // Same-reference pairs are not evaluated through these intermediates: orthonormal
                // parents use `DeviceOrthogonal`, and the host serves the rest.
                parents.push(x != w);
                if x == w {
                    pairs.push(DeviceWicksPair::default());
                    continue;
                }

                let idx = x * nref + w;
                let branches = &view.branches[idx];
                pairs.push(DeviceWicksPair {
                    evaluated: 1,
                    start: combinations.len() as u64,
                    na: branches.na as u64,
                    nb: branches.nb as u64,
                });

                for ia in 0..branches.na {
                    for ib in 0..branches.nb {
                        let pair = view.combination(idx, ia, ib, false);
                        let (ma, mb) = (pair.aa.m, pair.bb.m);
                        let mut combination = DeviceWicksCombination::default();

                        for (record, spin) in combination.spin.iter_mut().zip([&pair.aa, &pair.bb])
                        {
                            record.m = spin.m as u64;
                            record.nmo = spin.nmo as u64;
                            record.nocc = spin.nocc as u64;
                            record.pref = (spin.phase * T::from_real(spin.tilde_s_prod)).re();
                            for mi in 0..2 {
                                record.x[mi] = push(spin.x_slice(mi));
                                record.y[mi] = push(spin.y_slice(mi));
                                record.f0f[mi] = spin.f0f[mi].re();
                                record.f0h[mi] = spin.f0h[mi].re();
                                for mj in 0..2 {
                                    record.ff[mi][mj] = push(spin.ff_t_slice(mi, mj));
                                    record.fh[mi][mj] = push(spin.fh_t_slice(mi, mj));
                                    for mk in 0..2 {
                                        record.v[mi][mj][mk] = push(spin.v_t_slice(mi, mj, mk));
                                    }
                                }
                            }
                            for (slot, &sum) in JSUM.iter().enumerate() {
                                if sum <= spin.m {
                                    record.j[slot] = push(spin.j_slice(slot));
                                }
                            }
                            if ma == 0 && mb == 0 {
                                record.hcol0 = push(spin.hcol0_t_slice());
                            }
                            for (v0, value) in record.v0.iter_mut().zip(&spin.v0) {
                                *v0 = value.re();
                            }
                        }

                        let (ab, mixed) = (&pair.ab, &mut combination.mixed);
                        mixed.nab = ab.n() as u64;
                        for m0 in 0..2 {
                            for m1 in 0..2 {
                                mixed.vab0[m0][m1] = ab.vab0[m0][m1].re();
                                mixed.vba0[m0][m1] = ab.vba0[m0][m1].re();
                                for m2 in 0..2 {
                                    mixed.vab[m0][m1][m2] = push(ab.vab_t_slice(m0, m1, m2));
                                    mixed.vba[m0][m1][m2] = push(ab.vba_t_slice(m0, m1, m2));
                                }
                            }
                        }
                        for ma0 in 0..2 {
                            for maz in 0..2 {
                                for mb0 in 0..2 {
                                    for mby in 0..2 {
                                        if ma0 + maz <= ma && mb0 + mby <= mb {
                                            mixed.ii[ma0][maz][mb0][mby] =
                                                push(ab.iiab_slice(ma0, maz, mb0, mby));
                                        }
                                    }
                                }
                            }
                        }

                        combinations.push(combination);
                    }
                }
            }
        }

        Ok(Self {
            nref,
            parents,
            pairs: stream.clone_htod(&pairs)?,
            combinations: stream.clone_htod(&combinations)?,
            slab: stream.clone_htod(&slab)?,
        })
    }
}
