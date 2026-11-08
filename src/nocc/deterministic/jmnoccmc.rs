// nocc/deterministic/jmnoccmc.rs
//! Deterministic imaginary-time propagation of Jeziorski-Monkhorst NOCC (JM-NOCCMC), the
//! noise-free counterpart of the stochastic propagation.
//!
//! Every nonorthogonal reference `|{}^x\Psi\rangle` carries its own cluster operator
//! `\hat T_x = \sum_\mu t_{x\mu}\hat\tau_{x\mu}`, with excitations up to the cluster rank in its
//! orbital frame, and the state-universal ansatz is
//!
//! `|\Psi_k\rangle = \sum_x c_x^{(k)}|\Phi_x\rangle, |\Phi_x\rangle = e^{\hat T_x}|{}^x\Psi\rangle.`
//!
//! The span `\mathcal M` of the correlated references is invariant under the Hamiltonian when
//! `\hat H|\Phi_y\rangle = \sum_x G_{xy}|\Phi_x\rangle`, with
//! `G_{xy} = \sum_z[(S^e)^{-1}]_{xz}\langle{}^z\Psi|\hat H|\Phi_y\rangle` and
//! `S^e_{zy} = \langle{}^z\Psi|\Phi_y\rangle`; the eigenvalues `E_k` of `G` are the state
//! energies. With the oblique projector
//! `\hat P = \sum_{xz}|\Phi_x\rangle[(S^e)^{-1}]_{xz}\langle{}^z\Psi|` onto `\mathcal M`, the
//! invariance residual and the amplitude residuals are
//!
//! `|W_x\rangle = (\hat I - \hat P)\hat H|\Phi_x\rangle, R_{x\mu} = \langle{}^x\Psi|\hat\tau_{x\mu}^\dagger e^{-\hat T_x}|W_x\rangle.`
//!
//! The excitor populations `n_{x\mu}` and reference populations `N_x`, with
//! `t_{x\mu} = n_{x\mu}/N_x`, are propagated exactly as the stochastic propagation samples them,
//! with `G` evaluated exactly at every step.
//!
//! # Projection and expansion spaces
//!
//! `\langle{}^x\Psi|\hat\tau_{x\mu}^\dagger e^{-\hat T_x}` contains only determinants of
//! `|{}^x\Psi\rangle` whose rank does not exceed that of `\mu`, so every residual needs the
//! projections of `\hat H|\Phi_y\rangle` and `|\Phi_y\rangle` onto the reference and its
//! excitations up to the cluster rank alone. Each `|\Phi_x\rangle` is expanded over the
//! determinants of `|{}^x\Psi\rangle` up to an expansion rank, every rank by default. Since
//! `\hat H` changes the rank by at most two, any expansion rank of at least the cluster rank plus
//! two keeps every single-reference term exact. The references have different orbitals, so every
//! determinant of one reference overlaps the projection rows of another, and a lower expansion
//! truncates the overlaps `\langle{}^x\Psi|\hat\tau_{x\mu}^\dagger e^{-\hat T_x}|\Phi_y\rangle`
//! between different references; it is exact only for the complete expansion. The Hamiltonian
//! and overlap are therefore built once as rectangular matrices between the projection space
//! (rows) and the expansion space (columns), whose rows and columns are ordered by reference and
//! then by excitation rank.
//!
//! # Null space
//!
//! A null of the Jacobian `J_{y\mu,x\nu} = \partial R_{y\mu}/\partial t_{x\nu}` is a variation
//! with `\delta R_{y\mu} = 0` for every `y, \mu`. Splitting each variation of the correlated
//! references as `|\delta\Phi_x\rangle = \sum_z C_{zx}|\Phi_z\rangle + |X_x\rangle`, with
//! `C_{zx} = \sum_y[(S^e)^{-1}]_{zy}\langle{}^y\Psi|\delta\Phi_x\rangle` and
//! `|X_x\rangle = (\hat I - \hat P)|\delta\Phi_x\rangle`, gives
//!
//! `|\delta W_x\rangle = \sum_z C_{zx}|W_z\rangle + \hat H_\perp|X_x\rangle - \sum_y G_{yx}|X_y\rangle, \hat H_\perp = (\hat I - \hat P)\hat H(\hat I - \hat P).`
//!
//! A type I null is internal mixing only. It needs the mixing to be representable at the
//! cluster rank (condition A) and its residual projection to vanish (condition B), and gives
//! the `M(M-1)` exact nulls of `M` references at the complete cluster limit, which truncation
//! generically removes. A type II null is an external response only:
//! `\sum_x c_x^{(k)}|\delta W_x\rangle = (\hat H_\perp - E_k)|X^{(k)}\rangle` vanishes when an
//! external level meets a target root, the intruder problem. A type III null mixes the two.
//! The diagnostics measure the character of each low singular vector of `J` by the external
//! fraction `\sum_x\||X_x\rangle\|^2 / \sum_x(\|\hat P|\delta\Phi_x\rangle\|^2 + \||X_x\rangle\|^2)`.
//!
//! # References
//! - B. Jeziorski and H. J. Monkhorst, *Phys. Rev. A* **24**, 1668 (1981).

// Standard library imports.
use std::collections::HashMap;
use std::fs::create_dir_all;
use std::ops::Range;

// External crate imports.
use ndarray::{Array1, Array2, ArrayView1, s};
use ndarray_linalg::{Eig, Inverse, SVD};
use rayon::prelude::*;

// Crate-root imports.
use crate::basis::apply_excitation;
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::input::JMNOCCMCOptions;
use crate::maths::parallel_matvec;
use crate::nocc::{jm_energies, reference_determinants};

// Parent/sibling imports.
use super::write::{
    print_jm_block, print_jm_energies, print_jm_null_space, print_jm_stability,
    print_jm_table_header, print_jm_table_row, write_jm_null_history,
};

/// Sign bit packed into the excitation index of a cluster connection.
const NEGATIVE: u32 = 1 << 31;

/// One excitation `\hat\tau_\mu = a^\dagger_{\mathrm{parts}} a_{\mathrm{holes}}` of a reference's
/// cluster operator, on combined alpha-then-beta occupation bits.
struct ClusterExcitation {
    /// Annihilated orbitals.
    holes: u128,
    /// Created orbitals.
    parts: u128,
    /// Local index of the determinant `\hat\tau_\mu|{}^x\Psi\rangle`.
    target: usize,
    /// Sign of `\hat\tau_\mu|{}^x\Psi\rangle` relative to that determinant.
    sign: f64,
}

/// Amplitude-independent action of a reference's cluster operator, stored by target: every
/// `(k, \mu, \pm)` with `\hat\tau_\mu|k\rangle = \pm|j\rangle` for target `j`, so that
/// `[\hat T v]_j = \sum \pm t_\mu v_k`.
struct ClusterConnections {
    /// Start of each target's connections, with a final end offset.
    offsets: Vec<usize>,
    /// Source determinant `k` of each connection.
    sources: Vec<u32>,
    /// Excitation `\mu` of each connection, with [`NEGATIVE`] set for a negative sign.
    excitations: Vec<u32>,
}

/// Determinants of one reference's orbital frame, ordered by excitation rank.
pub struct ReferenceBlock {
    /// Columns of the block in the expansion space.
    columns: Range<usize>,
    /// First row of the block in the projection space.
    row: usize,
    /// Number of leading determinants of rank at most the cluster rank, which form the
    /// block's projection rows.
    projected: usize,
    /// Highest excitation rank in the block.
    rank: usize,
    /// Number of determinants in the block.
    size: usize,
    /// Excitations of the cluster operator.
    excitations: Vec<ClusterExcitation>,
    /// Action of the cluster operator on the block.
    connections: ClusterConnections,
}

/// Projection and expansion spaces of every reference.
pub struct JMSpace {
    /// Determinants of every expansion space, ordered by reference and then by rank.
    pub space: NOCISpace<f64>,
    /// Reference blocks.
    pub blocks: Vec<ReferenceBlock>,
}

impl JMSpace {
    /// Global determinant indices of the projection rows.
    /// # Arguments:
    /// - `self`: JM-NOCCMC spaces.
    /// # Returns:
    /// - `Vec<NOCIIndex>`: Projection-space determinants in row order.
    pub fn rows(&self) -> Vec<NOCIIndex> {
        self.blocks
            .iter()
            .flat_map(|b| (b.columns.start..b.columns.start + b.projected).map(NOCIIndex))
            .collect()
    }
}

/// Rectangular Hamiltonian and overlap between the projection and expansion spaces.
pub struct JMEquations {
    /// Reference blocks.
    blocks: Vec<ReferenceBlock>,
    /// Hamiltonian `\langle i|\hat H|j\rangle`, projection rows by expansion columns.
    h: Array2<f64>,
    /// Overlap `\langle i|j\rangle`, projection rows by expansion columns.
    s: Array2<f64>,
    /// Overlap between every pair of expansion columns, built only for the null-space
    /// diagnostics.
    metric: Option<Array2<f64>>,
}

/// Correlated references `|\Phi_x\rangle = e^{\hat T_x}|{}^x\Psi\rangle` at one set of
/// amplitudes, expanded over their blocks and projected, with `\hat H|\Phi_x\rangle`, onto every
/// projection row.
pub struct CorrelatedReferences {
    /// Expansion of every `|\Phi_x\rangle` over the determinants of its block.
    phi: Vec<Array1<f64>>,
    /// Projections `\langle k|\hat H|\Phi_x\rangle` onto every projection row.
    hphi: Vec<Array1<f64>>,
    /// Projections `\langle k|\Phi_x\rangle` onto every projection row.
    sphi: Vec<Array1<f64>>,
}

/// Effective overlap and Hamiltonian of the correlated references.
pub struct Effective {
    /// Effective overlap `S^e_{zx} = \langle{}^z\Psi|\Phi_x\rangle`.
    pub se: Array2<f64>,
    /// Reference projections `H^e_{zx} = \langle{}^z\Psi|\hat H|\Phi_x\rangle`.
    pub he: Array2<f64>,
    /// Effective Hamiltonian `G = (S^e)^{-1}H^e`.
    pub g: Array2<f64>,
}

/// Tabulate the action of a cluster operator on a block by target determinant. Each target
/// `j` is de-excited by every `\hat\tau_\mu`, giving the source
/// `k = (j \setminus \mathrm{parts}) \cup \mathrm{holes}` whenever the particles are occupied in
/// `j` and the holes are not, and the connection is kept if `k` lies in the block. The
/// excitations raise the rank, so every source precedes its target in the rank-ordered block.
/// # Arguments:
/// - `occupations`: Combined occupation of each local determinant.
/// - `lookup`: Occupation-to-local-index lookup.
/// - `excitations`: Excitations of the cluster operator.
/// # Returns:
/// - `ClusterConnections`: Connections of every target.
fn cluster_connections(
    occupations: &[u128],
    lookup: &HashMap<u128, usize>,
    excitations: &[ClusterExcitation],
) -> ClusterConnections {
    let targets = occupations
        .par_iter()
        .map(|&occupation| {
            excitations
                .iter()
                .enumerate()
                .filter_map(|(mu, e)| {
                    if occupation & e.parts != e.parts || occupation & !e.parts & e.holes != 0 {
                        return None;
                    }
                    let source = (occupation & !e.parts) | e.holes;
                    let k = *lookup.get(&source)?;
                    let (_, sign) = apply_excitation(source, e.holes, e.parts)?;
                    let packed = if sign < 0.0 {
                        mu as u32 | NEGATIVE
                    } else {
                        mu as u32
                    };
                    Some((k as u32, packed))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut offsets = Vec::with_capacity(targets.len() + 1);
    offsets.push(0);
    for target in &targets {
        offsets.push(offsets.last().unwrap() + target.len());
    }
    let (sources, excitations) = targets.into_iter().flatten().unzip();

    ClusterConnections {
        offsets,
        sources,
        excitations,
    }
}

/// Build the projection and expansion spaces of every reference. The expansion space of
/// reference `x` holds its determinants up to `expansion_rank`, every rank when `None`; its
/// leading determinants up to `cluster_rank` form the projection rows and define the cluster
/// excitations `\hat\tau_{x\mu}`.
/// # Arguments:
/// - `references`: Reference space, one determinant per reference.
/// - `options`: JM-NOCCMC options.
/// # Returns:
/// - `JMSpace`: Projection and expansion spaces.
pub fn jm_space(
    references: &NOCISpace<f64>,
    options: &JMNOCCMCOptions,
) -> JMSpace {
    let mut space = NOCISpace::from_parents(references.parents().to_vec());
    let mut blocks = Vec::with_capacity(references.len());
    let mut row = 0;

    for z in 0..references.len() {
        let parent = references.state(NOCIIndex(z)).parent;
        let norb = references.parents()[parent].ca.ncols();
        let (oa, ob) = references.occupations(NOCIIndex(z));
        let maximum = (oa.count_ones() + ob.count_ones()) as usize;
        let rank = options.expansion_rank.unwrap_or(maximum).min(maximum);
        let determinants = reference_determinants(oa, ob, norb, rank);

        let start = space.len();
        let mut occupations = Vec::with_capacity(determinants.len());
        for (k, &(da, db, _)) in determinants.iter().enumerate() {
            space.push(parent, da, db, format!("ref {z} det {k}"));
            occupations.push(da | (db << norb));
        }
        let lookup = occupations
            .iter()
            .enumerate()
            .map(|(k, &o)| (o, k))
            .collect::<HashMap<_, _>>();

        // Every excitation up to the cluster rank, with \tau_\mu|x> = sign |D_\mu>.
        let reference = occupations[0];
        let projected = determinants
            .iter()
            .take_while(|d| d.2 <= options.cluster_rank)
            .count();
        let excitations = (1..projected)
            .map(|k| {
                let holes = reference & !occupations[k];
                let parts = occupations[k] & !reference;
                let (_, sign) = apply_excitation(reference, holes, parts).unwrap();
                ClusterExcitation {
                    holes,
                    parts,
                    target: k,
                    sign,
                }
            })
            .collect::<Vec<_>>();
        let connections = cluster_connections(&occupations, &lookup, &excitations);
        print_jm_block(
            z,
            occupations.len(),
            projected,
            excitations.len(),
            connections.sources.len(),
        );

        blocks.push(ReferenceBlock {
            columns: start..space.len(),
            row,
            projected,
            rank,
            size: occupations.len(),
            excitations,
            connections,
        });
        row += projected;
    }

    JMSpace { space, blocks }
}

impl ReferenceBlock {
    /// Project a block vector onto the cluster excitations,
    /// `\langle{}^x\Psi|\hat\tau_{x\mu}^\dagger|v\rangle = \pm v_{D_\mu}` with the sign of
    /// `\hat\tau_{x\mu}|{}^x\Psi\rangle = \pm|D_\mu\rangle`.
    /// # Arguments:
    /// - `self`: Reference block.
    /// - `v`: Block vector covering at least the projection rows.
    /// # Returns:
    /// - `Array1<f64>`: One element per cluster excitation.
    fn project(
        &self,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        Array1::from_iter(self.excitations.iter().map(|e| e.sign * v[e.target]))
    }

    /// Expand an amplitude vector into the block, `\sum_\mu a_\mu\hat\tau_{x\mu}|{}^x\Psi\rangle`.
    /// # Arguments:
    /// - `self`: Reference block.
    /// - `a`: One element per cluster excitation.
    /// # Returns:
    /// - `Array1<f64>`: Block vector over the projection rows.
    fn excite(
        &self,
        a: &Array1<f64>,
    ) -> Array1<f64> {
        let mut v = Array1::<f64>::zeros(self.projected);
        for (e, &ai) in self.excitations.iter().zip(a) {
            v[e.target] = e.sign * ai;
        }
        v
    }

    /// Extend a block vector over the projection rows by zeros to the whole block.
    /// # Arguments:
    /// - `self`: Reference block.
    /// - `v`: Block vector over the projection rows.
    /// # Returns:
    /// - `Array1<f64>`: Block vector over every determinant of the block.
    fn extend(
        &self,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.size);
        out.slice_mut(s![..v.len()]).assign(v);
        out
    }
}

/// Apply `scale\,\hat T` to a block vector, `[scale\,\hat T v]_j = scale \sum \pm t_\mu v_k`
/// over the tabulated connections of each target `j`, in parallel over targets. The vector may
/// cover only a rank-ordered prefix of the block, since every source precedes its target.
/// # Arguments:
/// - `block`: Reference block.
/// - `t`: Amplitudes of the block's cluster operator.
/// - `v`: Block vector over a rank-ordered prefix of the block.
/// - `scale`: Prefactor of `\hat T`.
/// # Returns:
/// - `Array1<f64>`: `scale\,\hat T v` over the same prefix.
fn apply_cluster(
    block: &ReferenceBlock,
    t: &Array1<f64>,
    v: &Array1<f64>,
    scale: f64,
) -> Array1<f64> {
    let c = &block.connections;
    let out = (0..v.len())
        .into_par_iter()
        .map(|j| {
            let mut acc = 0.0;
            for e in c.offsets[j]..c.offsets[j + 1] {
                let packed = c.excitations[e];
                let term = t[(packed & !NEGATIVE) as usize] * v[c.sources[e] as usize];
                if packed & NEGATIVE == 0 {
                    acc += term;
                } else {
                    acc -= term;
                }
            }
            scale * acc
        })
        .collect::<Vec<_>>();
    Array1::from_vec(out)
}

/// Apply `e^{\pm\hat T}` to a block vector by its terminating series: every application of
/// `\hat T` raises the rank, so at most `rank` terms survive.
/// # Arguments:
/// - `block`: Reference block.
/// - `t`: Amplitudes of the block's cluster operator.
/// - `v`: Block vector over a rank-ordered prefix of the block.
/// - `sign`: `+1` for `e^{\hat T}` or `-1` for `e^{-\hat T}`.
/// # Returns:
/// - `Array1<f64>`: `e^{\pm\hat T}v` over the same prefix.
fn apply_exponential(
    block: &ReferenceBlock,
    t: &Array1<f64>,
    v: &Array1<f64>,
    sign: f64,
) -> Array1<f64> {
    let mut out = v.clone();
    let mut term = v.clone();
    for n in 1..=block.rank {
        term = apply_cluster(block, t, &term, sign / n as f64);
        out += &term;
    }
    out
}

impl JMEquations {
    /// Assemble JM-NOCCMC from its spaces and the rectangular matrices built over them.
    /// # Arguments:
    /// - `space`: Projection and expansion spaces.
    /// - `h`: Hamiltonian, projection rows by expansion columns.
    /// - `s`: Overlap, projection rows by expansion columns.
    /// - `metric`: Optional overlap between every pair of expansion columns, needed for the
    ///   external fractions of the null-space diagnostics.
    /// # Returns:
    /// - `Self`: Propagation data.
    pub fn new(
        space: JMSpace,
        h: Array2<f64>,
        s: Array2<f64>,
        metric: Option<Array2<f64>>,
    ) -> Self {
        Self {
            blocks: space.blocks,
            h,
            s,
            metric,
        }
    }

    /// Expand every correlated reference `|\Phi_x\rangle = e^{\hat T_x}|{}^x\Psi\rangle` over its
    /// block and project it, with `\hat H|\Phi_x\rangle`, onto every projection row.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference.
    /// # Returns:
    /// - `CorrelatedReferences`: Expansions and projections of the correlated references.
    pub fn correlate(
        &self,
        t: &[Array1<f64>],
    ) -> CorrelatedReferences {
        let phi = self
            .blocks
            .iter()
            .zip(t)
            .map(|(block, tx)| {
                let mut reference = Array1::<f64>::zeros(block.size);
                reference[0] = 1.0;
                apply_exponential(block, tx, &reference, 1.0)
            })
            .collect::<Vec<_>>();
        let (hphi, sphi) = self
            .blocks
            .iter()
            .zip(&phi)
            .map(|(block, phix)| {
                let cols = s![.., block.columns.clone()];
                (
                    parallel_matvec(&self.h.slice(cols), phix),
                    parallel_matvec(&self.s.slice(cols), phix),
                )
            })
            .unzip();
        CorrelatedReferences { phi, hphi, sphi }
    }

    /// Effective overlap and Hamiltonian from the reference rows of the projections,
    /// `S^e_{zx} = \langle{}^z\Psi|\Phi_x\rangle`, `H^e_{zx} = \langle{}^z\Psi|\hat H|\Phi_x\rangle`
    /// and `G = (S^e)^{-1}H^e`.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `phi`: Correlated references.
    /// # Returns:
    /// - `Effective`: Effective overlap and Hamiltonian.
    pub fn effective(
        &self,
        phi: &CorrelatedReferences,
    ) -> Effective {
        let m = self.blocks.len();
        let se = Array2::from_shape_fn((m, m), |(z, x)| phi.sphi[x][self.blocks[z].row]);
        let he = Array2::from_shape_fn((m, m), |(z, x)| phi.hphi[x][self.blocks[z].row]);
        let g = se.inv().unwrap().dot(&he);
        Effective { se, he, g }
    }

    /// Transformed invariance residuals `|Z_y\rangle = e^{-\hat T_y}|W_y\rangle` on the projection
    /// rows of every reference, with `|W_y\rangle = \hat H|\Phi_y\rangle - \sum_x G_{xy}|\Phi_x\rangle`.
    /// The `x = y` term is left out: it contributes `G_{yy}|{}^y\Psi\rangle`, on the reference row
    /// that no residual reads.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference.
    /// - `phi`: Correlated references at `t`.
    /// - `effective`: Effective overlap and Hamiltonian at `t`.
    /// # Returns:
    /// - `Vec<Array1<f64>>`: `|Z_y\rangle` of every reference, except on its reference row.
    pub fn transformed(
        &self,
        t: &[Array1<f64>],
        phi: &CorrelatedReferences,
        effective: &Effective,
    ) -> Vec<Array1<f64>> {
        let m = self.blocks.len();
        self.blocks
            .iter()
            .enumerate()
            .map(|(y, block)| {
                let rows = block.row..block.row + block.projected;
                let mut a = phi.hphi[y].slice(s![rows.clone()]).to_owned();
                for x in (0..m).filter(|&x| x != y) {
                    a.scaled_add(-effective.g[(x, y)], &phi.sphi[x].slice(s![rows.clone()]));
                }
                apply_exponential(block, &t[y], &a, -1.0)
            })
            .collect()
    }

    /// Amplitude residuals `R_{y\mu} = \langle{}^y\Psi|\hat\tau_{y\mu}^\dagger|Z_y\rangle`.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `z`: Transformed invariance residuals of every reference.
    /// # Returns:
    /// - `Vec<Array1<f64>>`: Residuals of every reference.
    pub fn residuals(
        &self,
        z: &[Array1<f64>],
    ) -> Vec<Array1<f64>> {
        self.blocks
            .iter()
            .zip(z)
            .map(|(block, zy)| block.project(zy))
            .collect()
    }

    /// Analytic Jacobian `J_{y\mu,x\nu} = \partial R_{y\mu} / \partial t_{x\nu}` over the
    /// flattened amplitudes, with the columns in parallel. Differentiating
    /// `R_{y\mu} = \langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}|W_y\rangle` with
    /// `\partial|\Phi_y\rangle/\partial t_{x\nu} = \delta_{yx}e^{\hat T_x}\hat\tau_{x\nu}|{}^x\Psi\rangle`
    /// gives
    /// `J_{y\mu,x\nu} = \langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}(\hat I-\hat P)(\delta_{yx}\hat H-G_{xy})e^{\hat T_x}\hat\tau_{x\nu}|{}^x\Psi\rangle - \delta_{yx}\langle{}^y\Psi|\hat\tau_{y\mu}^\dagger\hat\tau_{y\nu}e^{-\hat T_y}|W_y\rangle`,
    /// whose second term vanishes at a solution. It is formed by differentiating `|Z_y\rangle`
    /// without its reference row through
    /// `\partial e^{-\hat T_y}/\partial t_{y\nu} = -\hat\tau_{y\nu}e^{-\hat T_y}` and
    /// `\delta G_{yx} = \sum_z[(S^e)^{-1}]_{yz}[\langle{}^z\Psi|\hat H|\delta\Phi_x\rangle - \sum_u G_{ux}\langle{}^z\Psi|\delta\Phi_u\rangle]`.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference.
    /// - `phi`: Correlated references at `t`.
    /// - `effective`: Effective overlap and Hamiltonian at `t`.
    /// - `z`: Transformed invariance residuals at `t`.
    /// # Returns:
    /// - `Array2<f64>`: Jacobian over the flattened amplitudes.
    fn jacobian(
        &self,
        t: &[Array1<f64>],
        phi: &CorrelatedReferences,
        effective: &Effective,
        z: &[Array1<f64>],
    ) -> Array2<f64> {
        let m = self.blocks.len();
        let g = &effective.g;
        let sinv = effective.se.inv().unwrap();
        let columns = self
            .blocks
            .iter()
            .enumerate()
            .flat_map(|(x, b)| (0..b.excitations.len()).map(move |nu| (x, nu)))
            .collect::<Vec<_>>();

        let columns = columns
            .into_par_iter()
            .map(|(x, nu)| {
                let block = &self.blocks[x];
                let mut unit = Array1::<f64>::zeros(block.excitations.len());
                unit[nu] = 1.0;

                // `|\delta\Phi_x\rangle = e^{\hat T_x}\hat\tau_{x\nu}|{}^x\Psi\rangle` and its
                // projections, with those of `\hat H|\delta\Phi_x\rangle`, onto every row.
                let dphi =
                    apply_exponential(block, &t[x], &block.extend(&block.excite(&unit)), 1.0);
                let cols = s![.., block.columns.clone()];
                let dh = self.h.slice(cols).dot(&dphi);
                let ds = self.s.slice(cols).dot(&dphi);

                // `\delta G = (S^e)^{-1}(\delta H^e - \delta S^e G)`, where only column `x` of
                // `\delta H^e` and `\delta S^e` is nonzero.
                let delta = Array2::from_shape_fn((m, m), |(z, u)| {
                    let row = self.blocks[z].row;
                    let dhe = if u == x { dh[row] } else { 0.0 };
                    dhe - ds[row] * g[(x, u)]
                });
                let dg = sinv.dot(&delta);

                // `\delta|Z_y\rangle` on the rows of every `y`.
                let mut column = Vec::with_capacity(t.iter().map(|ty| ty.len()).sum());
                for (y, by) in self.blocks.iter().enumerate() {
                    let rows = by.row..by.row + by.projected;
                    let mut da = if y == x {
                        dh.slice(s![rows.clone()]).to_owned()
                    } else {
                        Array1::zeros(by.projected)
                    };
                    for u in (0..m).filter(|&u| u != y) {
                        da.scaled_add(-dg[(u, y)], &phi.sphi[u].slice(s![rows.clone()]));
                        if u == x {
                            da.scaled_add(-g[(x, y)], &ds.slice(s![rows.clone()]));
                        }
                    }
                    let mut dz = apply_exponential(by, &t[y], &da, -1.0);
                    if y == x {
                        dz -= &apply_cluster(by, &unit, &z[y], 1.0);
                    }
                    column.extend(by.project(&dz));
                }
                Array1::from_vec(column)
            })
            .collect::<Vec<_>>();

        let n = columns.len();
        let mut jac = Array2::<f64>::zeros((n, n));
        for (j, column) in columns.iter().enumerate() {
            jac.column_mut(j).assign(column);
        }
        jac
    }

    /// External fraction of an amplitude variation `\delta t`. The variation of every
    /// correlated reference, `|\delta\Phi_x\rangle = e^{\hat T_x}\delta\hat T_x|{}^x\Psi\rangle`, is
    /// split into its oblique projection onto the span `\mathcal M` of the correlated references
    /// and the complement,
    /// `|\delta\Phi_x\rangle = \sum_z C_{zx}|\Phi_z\rangle + |X_x\rangle` with
    /// `C_{zx} = \sum_y[(S^e)^{-1}]_{zy}\langle{}^y\Psi|\delta\Phi_x\rangle`, and the fraction is
    /// `\sum_x\||X_x\rangle\|^2 / \sum_x(\|\hat P|\delta\Phi_x\rangle\|^2 + \||X_x\rangle\|^2)`, zero
    /// for internal mixing only (type I) and one for an external response only (type II).
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference.
    /// - `phi`: Correlated references at `t`.
    /// - `effective`: Effective overlap and Hamiltonian at `t`.
    /// - `dt`: Flattened amplitude variation `\delta t`.
    /// # Returns:
    /// - `Option<f64>`: External fraction, or `None` without the expansion-column overlap.
    fn external_fraction(
        &self,
        t: &[Array1<f64>],
        phi: &CorrelatedReferences,
        effective: &Effective,
        dt: ArrayView1<f64>,
    ) -> Option<f64> {
        let metric = self.metric.as_ref()?;
        let sinv = effective.se.inv().unwrap();
        let norm = |v: &Array1<f64>| v.dot(&metric.dot(v));

        let (mut internal, mut external) = (0.0, 0.0);
        let mut start = 0;
        for (x, block) in self.blocks.iter().enumerate() {
            let len = block.excitations.len();
            let dtx = dt.slice(s![start..start + len]).to_owned();
            start += len;

            // `|\delta\Phi_x\rangle` over the expansion columns and `C_{zx}`.
            let dphix = apply_exponential(block, &t[x], &block.extend(&block.excite(&dtx)), 1.0);
            let mut dphi = Array1::<f64>::zeros(metric.nrows());
            dphi.slice_mut(s![block.columns.clone()]).assign(&dphix);
            let projections = Array1::from_iter(
                self.blocks
                    .iter()
                    .map(|by| self.s.slice(s![by.row, block.columns.clone()]).dot(&dphix)),
            );
            let c = sinv.dot(&projections);

            // `\hat P|\delta\Phi_x\rangle = \sum_z C_{zx}|\Phi_z\rangle` and `|X_x\rangle`.
            let mut inside = Array1::<f64>::zeros(metric.nrows());
            for (z, bz) in self.blocks.iter().enumerate() {
                inside
                    .slice_mut(s![bz.columns.clone()])
                    .scaled_add(c[z], &phi.phi[z]);
            }
            let outside = &dphi - &inside;
            internal += norm(&inside);
            external += norm(&outside);
        }
        Some(external / (internal + external))
    }

    /// Diagnose the Jacobian at one set of amplitudes, as deterministic NOCIQMC diagnoses its
    /// overlap null space: the stability of the linearised update from the eigenvalues of `J`,
    /// the partition of the singular values `\sigma_i` of `J` into a near-null part
    /// `\sigma_i < \epsilon\sigma_{\max}` and its complement, and the lowest singular values with
    /// the external fraction of their right singular vectors.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference.
    /// - `options`: JM-NOCCMC options.
    /// - `label`: Where in the propagation the diagnostics are taken.
    /// # Returns:
    /// - `(Array1<f64>, Array2<f64>)`: Singular values in descending order and the right
    ///   singular vectors as rows.
    fn null_diagnostics(
        &self,
        t: &[Array1<f64>],
        options: &JMNOCCMCOptions,
        label: &str,
    ) -> (Array1<f64>, Array2<f64>) {
        let phi = self.correlate(t);
        let effective = self.effective(&phi);
        let z = self.transformed(t, &phi, &effective);
        let jac = self.jacobian(t, &phi, &effective, &z);
        let (w, _) = jac.eig().unwrap();
        let mut w = w.to_vec();
        w.sort_by(|a, b| a.re.total_cmp(&b.re));
        let (_, sigma, vt) = jac.svd(false, true).unwrap();
        let vt = vt.unwrap();
        let (_, sse, _) = effective.se.svd(false, false).unwrap();

        // Lowest singular values with the external fractions of their right singular vectors.
        let n = sigma.len();
        let lowest = (0..options.null_n.min(n))
            .map(|k| {
                let i = n - 1 - k;
                (
                    sigma[i],
                    self.external_fraction(t, &phi, &effective, vt.row(i)),
                )
            })
            .collect::<Vec<_>>();

        print_jm_stability(label, &w, sse[0] / sse[sse.len() - 1]);
        print_jm_null_space(options.null_eps, &sigma, &lowest);
        (sigma, vt)
    }
}

/// Reference and excitor populations at the end of a deterministic JM-NOCCMC propagation.
pub struct JMPopulations {
    /// Amplitudes `t_{x\mu} = n_{x\mu}/N_x` of every reference.
    pub t: Vec<Array1<f64>>,
    /// Reference populations `N_x`, normalised to `\sum_x|N_x| = 1`.
    pub nref: Array1<f64>,
}

/// Deterministically propagate the JM-NOCCMC populations, the noise-free counterpart of the
/// stochastic propagation: the same populations, update and shift, with `G` evaluated exactly
/// at every step instead of being estimated from sampled report blocks. Excitor populations
/// `n_{x\mu}` and reference populations `N_x` carry the amplitudes `t_{x\mu} = n_{x\mu}/N_x`, and
/// with `g_x = \sum_y G_{xy}N_y/N_x` every step applies
/// `n_{x\mu} \leftarrow n_{x\mu} - \Delta\tau N_xR_{x\mu} - \Delta\tau(g_x - E_s)n_{x\mu}` and
/// `N_x \leftarrow N_x - \Delta\tau(g_x - E_s)N_x`, so that the reference populations follow the
/// power iteration of `G` towards its ground-state coefficients `c_x^{(0)}` and the amplitudes
/// follow `t_{x\mu} \leftarrow t_{x\mu} - \Delta\tau R_{x\mu}/[1 - \Delta\tau(g_x - E_s)]`, the plain
/// update once the shift balances the growth. The populations start from the reference NOCI
/// coefficients with no excitors, and are renormalised to
/// `\sum_x|N_x| + \sum_{x\mu}|n_{x\mu}| = 1` after every step with the removed amplitude kept,
/// from which the optional dynamic shift `E_s \leftarrow E_s - (\alpha/\Delta\tau)\Delta\ln N_w`
/// holds the population steady, as in deterministic NOCIQMC. The projected energy onto the NOCI
/// trial `\langle\Psi_T| = \sum_z c_z\langle{}^z\Psi|` is
/// `E = \sum_{zx}c_zH^e_{zx}N_x / \sum_{zx}c_zS^e_{zx}N_x`, as reported by the stochastic
/// propagation. Propagation stops when the largest residual falls below `residual_tol` and the
/// projected energy changes by less than `e_tol`. With `diagnostics`, the Jacobian at the
/// initial amplitudes is diagnosed and the amplitudes of every step are kept.
/// # Arguments:
/// - `jm`: Propagation data.
/// - `options`: JM-NOCCMC options.
/// - `reference`: Reference NOCI coefficients and energy, giving the initial reference
///   populations, the trial and the initial shift.
/// - `history`: Storage for the flattened amplitudes of every step when `diagnostics` is set.
/// # Returns:
/// - `Option<JMPopulations>`: Final populations, or `None` if the propagation diverged or did
///   not converge.
pub fn propagate_populations(
    jm: &JMEquations,
    options: &JMNOCCMCOptions,
    reference: (&[f64], f64),
    history: &mut Vec<Array1<f64>>,
) -> Option<JMPopulations> {
    let (c0, mut shift) = reference;
    let m = jm.blocks.len();
    let dt = options.dt;

    // Trial and initial reference populations from the normalised NOCI coefficients.
    let norm = c0.iter().map(|c| c.abs()).sum::<f64>();
    let trial = Array1::from_shape_fn(m, |z| {
        if c0.len() == m && norm > 0.0 {
            c0[z] / norm
        } else {
            1.0 / m as f64
        }
    });
    let mut nref = trial.clone();
    let mut n = jm
        .blocks
        .iter()
        .map(|b| Array1::<f64>::zeros(b.excitations.len()))
        .collect::<Vec<_>>();
    let mut logpop = 0.0_f64;
    let mut previous = f64::NAN;

    for iter in 0..options.max_steps {
        let t = n
            .iter()
            .zip(&nref)
            .map(|(nx, &nx0)| nx / nx0)
            .collect::<Vec<_>>();
        let phi = jm.correlate(&t);
        let effective = jm.effective(&phi);
        let residual = jm.residuals(&jm.transformed(&t, &phi, &effective));
        let rmax = residual
            .iter()
            .flatten()
            .fold(0.0_f64, |m, r| m.max(r.abs()));

        // A diverged step leaves G non-finite, which the eigensolver rejects.
        if !rmax.is_finite() || !effective.g.iter().all(|g| g.is_finite()) {
            println!("JM-NOCCMC propagation diverged at step {}.", iter);
            return None;
        }
        if options.diagnostics {
            if iter == 0 {
                jm.null_diagnostics(&t, options, "initial");
            }
            history.push(t.iter().flatten().copied().collect());
        }
        if iter == 0 {
            print_jm_table_header();
        }

        // Projected energy on the NOCI trial and the lowest state energy of `G`.
        let e = trial.dot(&effective.he.dot(&nref)) / trial.dot(&effective.se.dot(&nref));
        let eg = jm_energies(&effective.g)[0].re;
        let de = (e - previous).abs();
        let total = nref
            .iter()
            .chain(n.iter().flatten())
            .map(|v| v.abs())
            .sum::<f64>();
        let nrefabs = nref.iter().map(|v| v.abs()).sum::<f64>();
        print_jm_table_row(
            iter,
            (e, de, eg),
            shift,
            (logpop.exp() * total, logpop.exp() * nrefabs),
            rmax,
        );
        previous = e;
        if rmax < options.residual_tol && de < options.e_tol {
            println!("JM-NOCCMC converged after {} steps.", iter);
            let nref = nref / nrefabs;
            return Some(JMPopulations { t, nref });
        }

        // `-\Delta\tau N_xR_{x\mu} - \Delta\tau(g_x - E_s)n_{x\mu}` on the excitors and
        // `-\Delta\tau(g_x - E_s)N_x` on the references, with `g_xN_x = (GN)_x`.
        let gn = effective.g.dot(&nref);
        for (x, (nx, rx)) in n.iter_mut().zip(&residual).enumerate() {
            let growth = gn[x] / nref[x] - shift;
            *nx *= 1.0 - dt * growth;
            nx.scaled_add(-dt * nref[x], rx);
        }
        nref.zip_mut_with(&gn, |nx, &gx| *nx -= dt * (gx - shift * *nx));

        // Renormalise, keeping the removed amplitude, and steady the population by the shift.
        let total = nref
            .iter()
            .chain(n.iter().flatten())
            .map(|v| v.abs())
            .sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            println!("JM-NOCCMC populations became invalid at step {}.", iter + 1);
            return None;
        }
        nref /= total;
        for nx in n.iter_mut() {
            *nx /= total;
        }
        if options.dynamic_shift {
            shift -= options.dynamic_shift_alpha / dt * total.ln();
        }
        logpop += total.ln();
    }

    println!(
        "JM-NOCCMC did not converge after {} steps.",
        options.max_steps
    );
    None
}

/// Report the state energies and reference populations at the end of the propagation and, when
/// requested, diagnose the Jacobian there and write the history of the amplitude error
/// `\mathbf t_k - \mathbf t_\star` resolved along the right singular vectors of `J` at the final
/// amplitudes `\mathbf t_\star`, as deterministic NOCIQMC writes its coefficients resolved into
/// the relevant and null subspaces of the overlap.
/// # Arguments:
/// - `jm`: Propagation data.
/// - `populations`: Final populations.
/// - `history`: Flattened amplitudes of every step.
/// - `options`: JM-NOCCMC options.
/// - `dir`: Directory receiving the `jacobian_history` file.
/// # Returns:
/// - `f64`: Ground-state energy, the lowest real part of the eigenvalues of `G`.
pub fn report_converged(
    jm: &JMEquations,
    populations: &JMPopulations,
    history: &[Array1<f64>],
    options: &JMNOCCMCOptions,
    dir: &str,
) -> f64 {
    let JMPopulations { t, nref } = populations;
    let effective = jm.effective(&jm.correlate(t));
    let energies = jm_energies(&effective.g);
    let norms = t.iter().map(|tx| tx.dot(tx).sqrt()).collect::<Vec<_>>();
    print_jm_energies(&energies, &effective.se, nref, &norms);

    if options.diagnostics {
        let (sigma, vt) = jm.null_diagnostics(t, options, "final");
        let tfinal = t.iter().flatten().copied().collect::<Array1<f64>>();
        create_dir_all(dir).unwrap();
        let path = format!("{}/{}", dir, "jacobian_history");
        println!("Writing Jacobian history to {}...", path);
        write_jm_null_history(&path, history, &tfinal, (&sigma, &vt), options.null_eps);
    }

    energies[0].re
}
