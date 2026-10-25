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
use itertools::Itertools;
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eig, Inverse, SVD};
use num_complex::Complex64;
use rayon::prelude::*;

// Crate-root imports.
use crate::basis::apply_excitation;
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::input::{JMNOCCMCOptions, JMPropagator};
use crate::maths::parallel_matvec;

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

/// Residuals, effective Hamiltonian and effective overlap at one set of amplitudes.
pub struct JMEvaluation {
    /// Residuals `R_{y\mu}` of every reference.
    pub residual: Vec<Array1<f64>>,
    /// Effective Hamiltonian `G = (S^e)^{-1}H^e`.
    pub g: Array2<f64>,
    /// Effective overlap `S^e_{zx} = \langle{}^z\Psi|\Phi_x\rangle`.
    pub se: Array2<f64>,
}

/// Enumerate the determinants of one reference up to an excitation rank, ordered by rank.
/// # Arguments:
/// - `oa`: Alpha occupation of the reference.
/// - `ob`: Beta occupation of the reference.
/// - `norb`: Number of spatial orbitals in the reference frame.
/// - `rank`: Highest excitation rank, capped at the largest possible rank.
/// # Returns:
/// - `Vec<(u128, u128, usize)>`: Alpha and beta occupations and excitation rank.
fn reference_determinants(
    oa: u128,
    ob: u128,
    norb: usize,
    rank: usize,
) -> Vec<(u128, u128, usize)> {
    let split = |occ: u128| {
        let (occupied, virtuals): (Vec<usize>, Vec<usize>) =
            (0..norb).partition(|&p| occ & (1u128 << p) != 0);
        (occupied, virtuals)
    };
    let (aocc, avir) = split(oa);
    let (bocc, bvir) = split(ob);
    let excite = |occ: u128, holes: &[usize], parts: &[usize]| {
        let occ = holes.iter().fold(occ, |o, &h| o & !(1u128 << h));
        parts.iter().fold(occ, |o, &p| o | (1u128 << p))
    };

    let mut out = Vec::new();
    for order in 0..=rank {
        for arank in 0..=order {
            let brank = order - arank;
            for aholes in aocc.iter().copied().combinations(arank) {
                for aparts in avir.iter().copied().combinations(arank) {
                    for bholes in bocc.iter().copied().combinations(brank) {
                        for bparts in bvir.iter().copied().combinations(brank) {
                            out.push((
                                excite(oa, &aholes, &aparts),
                                excite(ob, &bholes, &bparts),
                                order,
                            ));
                        }
                    }
                }
            }
        }
    }
    out
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

        // Projections of H|\Phi_x> and |\Phi_x> onto every projection row.
        let (hphi, sphi): (Vec<_>, Vec<_>) = self
            .blocks
            .iter()
            .zip(t)
            .map(|(block, tx)| {
                let mut reference = Array1::<f64>::zeros(block.size);
                reference[0] = 1.0;
                let phi = apply_exponential(block, tx, &reference, 1.0);
                let cols = s![.., block.columns.clone()];
                (
                    parallel_matvec(&self.h.slice(cols), &phi),
                    parallel_matvec(&self.s.slice(cols), &phi),
                )
            })
            .unzip();

        let se = Array2::from_shape_fn((m, m), |(z, x)| sphi[x][self.blocks[z].row]);
        let he = Array2::from_shape_fn((m, m), |(z, x)| hphi[x][self.blocks[z].row]);
        let g = se.inv().unwrap().dot(&he);

        let residual = self
            .blocks
            .iter()
            .enumerate()
            .map(|(y, block)| {
                let rows = block.row..block.row + block.projected;
                let mut a = hphi[y].slice(s![rows.clone()]).to_owned();
                for x in (0..m).filter(|&x| x != y) {
                    a.scaled_add(-g[(x, y)], &sphi[x].slice(s![rows.clone()]));
                }
                let projected = apply_exponential(block, &t[y], &a, -1.0);
                Array1::from_iter(
                    block
                        .excitations
                        .iter()
                        .map(|e| e.sign * projected[e.target]),
                )
            })
            .collect();

        JMEvaluation { residual, g, se }
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

    /// Finite-difference Jacobian `J_{y\mu,x\nu} = \partial R_{y\mu} / \partial t_{x\nu}` by
    /// central differences over the flattened amplitudes, with the columns in parallel.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `t`: Amplitudes of every reference.
    /// - `step`: Finite-difference step.
    /// # Returns:
    /// - `Array2<f64>`: Jacobian over the flattened amplitudes.
    fn jacobian(
        &self,
        t: &[Array1<f64>],
        step: f64,
    ) -> Array2<f64> {
        let flat = t.iter().flatten().copied().collect::<Array1<f64>>();
        let columns = (0..flat.len())
            .into_par_iter()
            .map(|j| {
                let mut plus = flat.clone();
                plus[j] += step;
                let mut minus = flat.clone();
                minus[j] -= step;
                let rp = self
                    .evaluate(&self.split(&plus))
                    .residual
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Array1<f64>>();
                let rm = self
                    .evaluate(&self.split(&minus))
                    .residual
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Array1<f64>>();
                (rp - rm) / (2.0 * step)
            })
            .collect::<Vec<_>>();
        let mut jac = Array2::<f64>::zeros((flat.len(), flat.len()));
        for (j, column) in columns.iter().enumerate() {
            jac.column_mut(j).assign(column);
        }
        jac
    }

    /// Split flattened amplitudes into those of every reference.
    /// # Arguments:
    /// - `self`: Propagation data.
    /// - `flat`: Flattened amplitudes.
    /// # Returns:
    /// - `Vec<Array1<f64>>`: Amplitudes of every reference.
    fn split(
        &self,
        flat: &Array1<f64>,
    ) -> Vec<Array1<f64>> {
        let mut start = 0;
        self.blocks
            .iter()
            .map(|b| {
                let len = b.excitations.len();
                start += len;
                flat.slice(s![start - len..start]).to_owned()
            })
            .collect()
    }
}

/// Eigenvalues of the effective Hamiltonian, sorted by real part.
/// # Arguments:
/// - `g`: Effective Hamiltonian.
/// # Returns:
/// - `Vec<Complex64>`: State energies.
pub fn jm_energies(g: &Array2<f64>) -> Vec<Complex64> {
    let (w, _) = g.eig().unwrap();
    let mut w = w.to_vec();
    w.sort_by(|a, b| a.re.total_cmp(&b.re));
    w
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
        let jac = jm.jacobian(t, 1e-5);
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
