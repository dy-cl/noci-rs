// nocc/deterministic/jmnoccmc.rs
//! Deterministic imaginary-time propagation of Jeziorski-Monkhorst NOCC (JM-NOCCMC).
//!
//! Every nonorthogonal reference `x` carries its own cluster operator
//! `\hat T_x = \sum_\mu t_{x\mu}\hat\tau_{x\mu}`, with excitations up to the cluster rank in the
//! orbital frame of `x`, and the state-universal ansatz is
//!
//! `|\Psi_k\rangle = \sum_x c_x^{(k)}|\Phi_x\rangle, |\Phi_x\rangle = e^{\hat T_x}|{}^x\Psi\rangle.`
//!
//! Projecting `\hat H|\Phi_x\rangle = \sum_y G_{yx}|\Phi_y\rangle` onto the references gives the
//! effective overlap and Hamiltonian `S^e_{zx} = \langle{}^z\Psi|\Phi_x\rangle` and
//! `H^e_{zx} = \langle{}^z\Psi|\hat H|\Phi_x\rangle`, the effective Hamiltonian
//! `G = (S^e)^{-1}H^e` whose eigenvalues are the state energies, and the residuals
//!
//! `R_{y\mu} = \langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}\hat H|\Phi_y\rangle - \sum_{x \neq y} G_{xy}\langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}|\Phi_x\rangle.`
//!
//! The amplitudes are propagated by `t \leftarrow t - \Delta\tau R` or by a damped Jacobi step.
//!
//! # Projection and expansion spaces
//!
//! `\langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}` contains only determinants of `y`
//! whose rank does not exceed that of `\mu`, so every residual needs the projections of
//! `\hat H|\Phi_x\rangle` and `|\Phi_x\rangle` onto the reference and its excitations up to the
//! cluster rank alone. Each `|\Phi_x\rangle` is expanded over the determinants of `x` up to an
//! expansion rank, every rank by default. Since `\hat H` changes the rank by at most two, any
//! expansion rank of at least the cluster rank plus two keeps every single-reference term
//! exact. The references have different orbitals, so every determinant of `x` overlaps the
//! projection rows of `y`, and a lower expansion truncates the overlaps
//! `\langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}|\Phi_x\rangle` between different
//! references; it is exact only for the complete expansion. The Hamiltonian and overlap are therefore built once as rectangular matrices
//! between the projection space (rows) and the expansion space (columns), whose rows and
//! columns are ordered by reference and then by excitation rank.
//!
//! # References
//! - B. Jeziorski and H. J. Monkhorst, *Phys. Rev. A* **24**, 1668 (1981).

// Standard library imports.
use std::collections::HashMap;
use std::ops::Range;

// External crate imports.
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eig, Inverse, SVD};
use rayon::prelude::*;

// Crate-root imports.
use crate::basis::apply_excitation;
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::input::{JMNOCCMCOptions, JMPropagator};
use crate::maths::parallel_matvec;
use crate::nocc::{jm_energies, reference_determinants};

// Parent/sibling imports.
use super::write::{
    print_jm_block, print_jm_energies, print_jm_stability, print_jm_table_header,
    print_jm_table_row,
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
    /// Diagonal denominators `D_{y\mu} = H_{\mu\mu} - H_{yy}` of each reference.
    denominators: Vec<Array1<f64>>,
}

/// Residuals, effective Hamiltonian and effective overlap at one set of amplitudes, with the
/// intermediates from which the Jacobian is formed.
pub struct JMEvaluation {
    /// Residuals `R_{y\mu}` of every reference.
    pub residual: Vec<Array1<f64>>,
    /// Effective Hamiltonian `G = (S^e)^{-1}H^e`.
    pub g: Array2<f64>,
    /// Effective overlap `S^e_{zx} = \langle{}^z\Psi|\Phi_x\rangle`.
    pub se: Array2<f64>,
    /// Projections `\langle k|\Phi_x\rangle` of every `|\Phi_x\rangle` onto every projection row.
    sphi: Vec<Array1<f64>>,
    /// `e^{-\hat T_y}(\hat H|\Phi_y\rangle - \sum_{x \neq y}G_{xy}|\Phi_x\rangle)` of every
    /// reference on its projection rows, equal to `e^{-\hat T_y}|W_y\rangle` except on the
    /// reference row.
    contracted: Vec<Array1<f64>>,
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
    /// # Returns:
    /// - `Self`: Propagation data with the diagonal denominators of every reference.
    pub fn new(
        space: JMSpace,
        h: Array2<f64>,
        s: Array2<f64>,
    ) -> Self {
        let denominators = space
            .blocks
            .iter()
            .map(|b| {
                let col = b.columns.start;
                let h00 = h[(b.row, col)];
                Array1::from_iter(
                    b.excitations
                        .iter()
                        .map(|e| h[(b.row + e.target, col + e.target)] - h00),
                )
            })
            .collect();

        Self {
            blocks: space.blocks,
            h,
            s,
            denominators,
        }
    }

    /// Evaluate the effective matrices and the residuals of every reference.
    /// `H|\Phi_x\rangle` and `|\Phi_x\rangle` are projected onto the projection rows, giving
    /// `S^e` and `H^e` from the reference rows and, for each reference `y`, the projections
    /// `a_k = \langle k|\hat H|\Phi_y\rangle - \sum_{x \neq y} G_{xy}\langle k|\Phi_x\rangle` of its
    /// rows, from which `R_{y\mu} = [e^{-\hat T_y}a]_\mu` since each block is orthonormal.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference.
    /// # Returns:
    /// - `JMEvaluation`: Residuals, effective Hamiltonian and effective overlap.
    pub fn evaluate(
        &self,
        t: &[Array1<f64>],
    ) -> JMEvaluation {
        let m = self.blocks.len();

        // Expansions of |\Phi_x> and their projections, with those of H|\Phi_x>, onto every
        // projection row.
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
        let (hphi, sphi): (Vec<_>, Vec<_>) = self
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

        let se = Array2::from_shape_fn((m, m), |(z, x)| sphi[x][self.blocks[z].row]);
        let he = Array2::from_shape_fn((m, m), |(z, x)| hphi[x][self.blocks[z].row]);
        let g = se.inv().unwrap().dot(&he);

        // `e^{-\hat T_y}|W_y\rangle` on the rows of `y`, with
        // `|W_y\rangle = \hat H|\Phi_y\rangle - \sum_x G_{xy}|\Phi_x\rangle`; the `x = y` term
        // contributes `G_{yy}|{}^y\Psi\rangle`, on the reference row that no residual reads.
        let contracted = self
            .blocks
            .iter()
            .enumerate()
            .map(|(y, block)| {
                let rows = block.row..block.row + block.projected;
                let mut a = hphi[y].slice(s![rows.clone()]).to_owned();
                for x in (0..m).filter(|&x| x != y) {
                    a.scaled_add(-g[(x, y)], &sphi[x].slice(s![rows.clone()]));
                }
                apply_exponential(block, &t[y], &a, -1.0)
            })
            .collect::<Vec<_>>();
        let residual = self
            .blocks
            .iter()
            .zip(&contracted)
            .map(|(block, z)| block.project(z))
            .collect();

        JMEvaluation {
            residual,
            g,
            se,
            sphi,
            contracted,
        }
    }

    /// Apply one amplitude update, `t_{y\mu} \leftarrow t_{y\mu} - \Delta\tau R_{y\mu}` or
    /// `t_{y\mu} \leftarrow t_{y\mu} - \alpha R_{y\mu} / D_{y\mu}`.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference, updated in place.
    /// - `residual`: Residuals at `t`.
    /// - `options`: JM-NOCCMC options.
    /// # Returns:
    /// - None.
    fn propagate_step(
        &self,
        t: &mut [Array1<f64>],
        residual: &[Array1<f64>],
        options: &JMNOCCMCOptions,
    ) {
        for ((ty, ry), dy) in t.iter_mut().zip(residual).zip(&self.denominators) {
            match options.propagator {
                JMPropagator::Plain => ty.scaled_add(-options.dt, ry),
                JMPropagator::Diagonal => *ty -= &(options.damping * ry / dy),
            }
        }
    }

    /// Analytic Jacobian `J_{y\mu,x\nu} = \partial R_{y\mu} / \partial t_{x\nu}` over the
    /// flattened amplitudes, with the columns in parallel. Differentiating
    /// `R_{y\mu} = \langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}|W_y\rangle` with
    /// `\partial|\Phi_y\rangle/\partial t_{x\nu} = \delta_{yx}e^{\hat T_x}\hat\tau_{x\nu}|{}^x\Psi\rangle`
    /// gives
    ///
    /// `J_{y\mu,x\nu} = \langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}(\hat I-\hat P)(\delta_{yx}\hat H-G_{xy})e^{\hat T_x}\hat\tau_{x\nu}|{}^x\Psi\rangle - \delta_{yx}\langle{}^y\Psi|\hat\tau_{y\mu}^\dagger\hat\tau_{y\nu}e^{-\hat T_y}|W_y\rangle,`
    ///
    /// whose second term vanishes at a solution. It is formed by differentiating
    /// `e^{-\hat T_y}(\hat H|\Phi_y\rangle - \sum_{x \neq y}G_{xy}|\Phi_x\rangle)`, which differs
    /// from `e^{-\hat T_y}|W_y\rangle` only on the reference row, through
    /// `\partial e^{-\hat T_y}/\partial t_{y\nu} = -\hat\tau_{y\nu}e^{-\hat T_y}` and
    /// `\delta G_{yx} = \sum_z[(S^e)^{-1}]_{yz}[\langle{}^z\Psi|\hat H|\delta\Phi_x\rangle - \sum_u G_{ux}\langle{}^z\Psi|\delta\Phi_u\rangle]`.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference.
    /// - `ev`: Evaluation at `t`.
    /// # Returns:
    /// - `Array2<f64>`: Jacobian over the flattened amplitudes.
    fn jacobian(
        &self,
        t: &[Array1<f64>],
        ev: &JMEvaluation,
    ) -> Array2<f64> {
        let m = self.blocks.len();
        let sinv = ev.se.inv().unwrap();
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
                    dhe - ds[row] * ev.g[(x, u)]
                });
                let dg = sinv.dot(&delta);

                // `\delta(e^{-\hat T_y}|W_y\rangle)` on the rows of every `y`.
                let mut column = Vec::with_capacity(t.iter().map(|ty| ty.len()).sum());
                for (y, by) in self.blocks.iter().enumerate() {
                    let rows = by.row..by.row + by.projected;
                    let mut da = if y == x {
                        dh.slice(s![rows.clone()]).to_owned()
                    } else {
                        Array1::zeros(by.projected)
                    };
                    for u in (0..m).filter(|&u| u != y) {
                        da.scaled_add(-dg[(u, y)], &ev.sphi[u].slice(s![rows.clone()]));
                        if u == x {
                            da.scaled_add(-ev.g[(x, y)], &ds.slice(s![rows.clone()]));
                        }
                    }
                    let mut dz = apply_exponential(by, &t[y], &da, -1.0);
                    if y == x {
                        dz -= &apply_cluster(by, &unit, &ev.contracted[y], 1.0);
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
}

/// Propagate the JM-NOCCMC amplitudes from zero until the largest residual falls below the
/// tolerance or the step limit is reached, printing one table row per step.
/// # Arguments:
/// - `jm`: Propagation data.
/// - `options`: JM-NOCCMC options.
/// # Returns:
/// - `Option<(Vec<Array1<f64>>, JMEvaluation)>`: Converged amplitudes and their evaluation, or
///   `None` if the propagation diverged or did not converge.
pub fn propagate_amplitudes(
    jm: &JMEquations,
    options: &JMNOCCMCOptions,
) -> Option<(Vec<Array1<f64>>, JMEvaluation)> {
    let mut t = jm
        .blocks
        .iter()
        .map(|b| Array1::zeros(b.excitations.len()))
        .collect::<Vec<_>>();
    let mut previous = f64::NAN;

    print_jm_table_header();
    for iter in 0..options.max_steps {
        let ev = jm.evaluate(&t);
        let rmax = ev
            .residual
            .iter()
            .flatten()
            .fold(0.0_f64, |m, r| m.max(r.abs()));
        let rnorm = ev.residual.iter().map(|r| r.dot(r)).sum::<f64>().sqrt();

        // A diverged step leaves G non-finite, which the eigensolver rejects.
        if !rnorm.is_finite() || !ev.g.iter().all(|g| g.is_finite()) {
            println!("JM-NOCCMC propagation diverged at step {}.", iter);
            return None;
        }

        let e = jm_energies(&ev.g)[0].re;
        let tnorm = t.iter().map(|x| x.dot(x)).sum::<f64>().sqrt();
        print_jm_table_row(iter, e, (e - previous).abs(), rmax, rnorm, tnorm);
        previous = e;
        if rmax < options.residual_tol {
            println!("JM-NOCCMC converged after {} steps.", iter);
            return Some((t, ev));
        }
        jm.propagate_step(&mut t, &ev.residual, options);
    }

    println!(
        "JM-NOCCMC did not converge after {} steps.",
        options.max_steps
    );
    None
}

/// Report the state energies at converged amplitudes and, when requested, the stability of the
/// linearised update. The plain step `t \leftarrow t - \Delta\tau R` linearises to
/// `I - \Delta\tau J`, which contracts only if every eigenvalue `\lambda` of `J` has
/// `\mathrm{Re}\,\lambda > 0` and `\Delta\tau < 2\,\mathrm{Re}\,\lambda / |\lambda|^2`; the
/// diagonal step linearises to `I - \alpha D^{-1}J` with the same conditions on `D^{-1}J`.
/// # Arguments:
/// - `jm`: Propagation data.
/// - `t`: Converged amplitudes.
/// - `ev`: Evaluation at `t`.
/// - `options`: JM-NOCCMC options.
/// # Returns:
/// - `f64`: Ground-state energy, the lowest real part of the eigenvalues of `G`.
pub fn report_converged(
    jm: &JMEquations,
    t: &[Array1<f64>],
    ev: &JMEvaluation,
    options: &JMNOCCMCOptions,
) -> f64 {
    let energies = jm_energies(&ev.g);
    let norms = t.iter().map(|tx| tx.dot(tx).sqrt()).collect::<Vec<_>>();
    print_jm_energies(&energies, &ev.se, &norms);

    if options.diagnostics {
        let jac = jm.jacobian(t, ev);
        let propagator = match options.propagator {
            JMPropagator::Plain => jac.clone(),
            JMPropagator::Diagonal => {
                let d = jm.denominators.iter().flatten().collect::<Vec<_>>();
                Array2::from_shape_fn(jac.dim(), |(i, j)| jac[(i, j)] / d[i])
            }
        };
        let (w, _) = propagator.eig().unwrap();
        let mut w = w.to_vec();
        w.sort_by(|a, b| a.re.total_cmp(&b.re));
        let (_, sv, _) = jac.svd(false, false).unwrap();
        let (_, sse, _) = ev.se.svd(false, false).unwrap();
        print_jm_stability(
            &w,
            sv.iter().copied().fold(f64::INFINITY, f64::min),
            sse[0] / sse[sse.len() - 1],
        );
    }

    energies[0].re
}
