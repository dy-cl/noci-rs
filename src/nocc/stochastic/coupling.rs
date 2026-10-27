// nocc/stochastic/coupling.rs
//! Sampling of the coupling between references.
//!
//! The coupling of block `y` to the correlated reference `x \ne y` enters the residual as
//!
//! `\langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}|\tilde\Phi_x\rangle = \sum_{\nu\subseteq\mu}\langle D_\mu|e^{-\hat T_y}|D_\nu\rangle\,q^x_y(\nu),\quad q^x_y(\nu) = \langle D_\nu|\tilde\Phi_x\rangle,`
//!
//! because `e^{-\hat T_y}` acting to the left only removes excitations. Unlike the
//! same-reference term this does not terminate: `e^{\hat T_x}` and `e^{-\hat T_y}` act in
//! different orbital frames, and the nonorthogonal overlap connects determinants of every rank.
//! The projection `q^x_y` onto the determinants of `y` up to the cluster rank is therefore
//! sampled from clusters of the unlinked expansion `e^{\hat T_x}` up to the expansion rank,
//! the reference row exactly per cluster and the remaining rows uniformly. The same clusters
//! give the effective overlap and Hamiltonian `\langle{}^y\Psi|\tilde\Phi_x\rangle` and
//! `\langle{}^y\Psi|\hat H|\tilde\Phi_x\rangle`. The left action of `e^{-\hat T_y}` on the sparse
//! projection is then evaluated exactly from the current amplitudes of `y`.

// External crate imports.
use rand::Rng;
use rayon::prelude::*;

// Crate-root imports.
use crate::determinant::{ExcitedDeterminant, ExcitedState};
use crate::qmc::{FastMap, QmcRng};

// Parent/sibling imports.
use super::population::{Block, Excitor, ExcitorKey, rank};
use super::state::CouplingBatchedRequest;

/// Number of non-reference projection rows drawn per cluster and coupled block.
const ROWDRAWS: usize = 4;

/// Sampled projections of one step: for every ordered pair `(x, y)` the effective overlap and
/// Hamiltonian contributions and the sparse projection `q^x_y(\nu)`.
pub(super) struct Projections {
    /// `\sum w\langle{}^y\Psi|D_C\rangle`, indexed `[y][x]`.
    pub(super) se: Vec<Vec<f64>>,
    /// `\sum w\langle{}^y\Psi|\hat H|D_C\rangle`, indexed `[y][x]`.
    pub(super) he: Vec<Vec<f64>>,
    /// `q^x_y(\nu)` keyed by the occupations of `\nu`, indexed `[y][x]`.
    pub(super) q: Vec<Vec<FastMap<(u128, u128), f64>>>,
}

impl Projections {
    /// Construct empty projections for `M` references.
    /// # Arguments:
    /// - `m`: Number of references.
    /// # Returns:
    /// - `Projections`: Zero projections.
    pub(super) fn new(m: usize) -> Self {
        Self {
            se: vec![vec![0.0; m]; m],
            he: vec![vec![0.0; m]; m],
            q: (0..m)
                .map(|_| (0..m).map(|_| FastMap::default()).collect())
                .collect(),
        }
    }

    /// Reset every projection to zero while retaining the map allocations.
    /// # Arguments:
    /// - `self`: Projections.
    /// # Returns:
    /// - `()`: Clears the projections in place.
    pub(super) fn clear(&mut self) {
        for ((se, he), q) in self.se.iter_mut().zip(&mut self.he).zip(&mut self.q) {
            se.iter_mut().for_each(|v| *v = 0.0);
            he.iter_mut().for_each(|v| *v = 0.0);
            q.iter_mut().for_each(FastMap::clear);
        }
    }
}

/// Queue the projection of one sampled cluster `w\,\phi_C|D_C\rangle` of block `x` onto every
/// other block `y`: the reference row exactly, giving `\langle{}^y\Psi|D_C\rangle` and
/// `\langle{}^y\Psi|\hat H|D_C\rangle`, and `ROWDRAWS` uniformly drawn rows `\nu` of rank
/// `1..=cluster_rank` with weight `n_{\text{rows}}/\text{ROWDRAWS}`.
/// # Arguments:
/// - `rows`: Prepared projection rows of every block with ranks `1..=cluster_rank`.
/// - `x`: Source block.
/// - `cluster`: Prepared collapsed determinant `|D_C\rangle` and signed weight `w\phi_C`.
/// - `rng`: Random-number generator.
/// - `queue`: Prepared cluster determinants and coupling requests of the task.
/// # Returns:
/// - `()`: Queues the cluster's requests.
pub(super) fn project_cluster(
    rows: &[Vec<ExcitedState>],
    x: usize,
    cluster: (ExcitedState, f64),
    rng: &mut QmcRng,
    queue: (&mut Vec<ExcitedState>, &mut Vec<CouplingBatchedRequest>),
) {
    let (det, weight) = cluster;
    let (clusters, requests) = queue;
    let index = clusters.len();
    clusters.push(det);
    for (y, rows) in rows.iter().enumerate() {
        if y == x {
            continue;
        }

        // Reference row: effective overlap and Hamiltonian.
        requests.push(CouplingBatchedRequest {
            cluster: index,
            x,
            y,
            row: None,
            coefficient: weight,
        });

        // Remaining rows drawn uniformly.
        if rows.is_empty() {
            continue;
        }
        let scale = weight * rows.len() as f64 / ROWDRAWS as f64;
        for _ in 0..ROWDRAWS {
            requests.push(CouplingBatchedRequest {
                cluster: index,
                x,
                y,
                row: Some(rng.gen_range(0..rows.len())),
                coefficient: scale,
            });
        }
    }
}

/// Apply `e^{-\hat T_y}` from the left to a projection, giving for every projectee
/// `\mu \ne {}^y\Psi` of rank at most the cluster rank
/// `\sum_\nu\langle D_\mu|e^{-\hat T_y}|D_\nu\rangle q(\nu)` with
/// `\langle D_\mu|e^{-\hat T_y}|D_\nu\rangle = \sum_K(-1)^{|K|}\prod_{k\in K}t_{yk}\langle D_\mu|\hat\tau_K|D_\nu\rangle`
/// over sets `K` of distinct excitors, since a repeated excitor vanishes and the `|K|!` orderings
/// cancel the factorial of the exponential. With the excitors distributed over `P` ranks, each
/// rank sums the sets of its own excitors with weight `P^{|K|-1}`, the inverse probability that
/// they share a rank, and only rank zero adds the empty set.
/// # Arguments:
/// - `block`: Population of block `y` on this rank.
/// - `index`: Block index `y`.
/// - `q`: Complete projection `q(\nu)` keyed by occupations.
/// - `cluster_rank`: Highest excitation rank of the cluster operator.
/// - `distribution`: Number of ranks `P` and whether this rank adds the empty set.
/// # Returns:
/// - `Vec<(ExcitorKey, f64)>`: Population changes per unit time.
pub(super) fn left_action(
    block: &Block,
    index: usize,
    q: &[((u128, u128), f64)],
    cluster_rank: usize,
    distribution: (f64, bool),
) -> Vec<(ExcitorKey, f64)> {
    let reference = &block.reference;
    let (ranks, empty) = distribution;
    // Excitors in increasing rank, so the enumeration stops at the first one over budget.
    let mut amplitudes = block
        .excitors
        .iter()
        .map(|&(excitor, n)| (excitor, ranks * n / block.nref))
        .collect::<Vec<(Excitor, f64)>>();
    amplitudes.sort_by_key(|(excitor, _)| rank(reference, &excitor.det));

    // Depth-first enumeration of sets of distinct excitors in increasing order; each excitor
    // carries `P t`, and the leading `1/P` gives the weight `P^{|K|-1}`.
    fn visit(
        det: ExcitedDeterminant,
        coefficient: f64,
        start: usize,
        context: (&ExcitedDeterminant, &[(Excitor, f64)], usize, usize),
        out: &mut Vec<(ExcitorKey, f64)>,
    ) {
        let (reference, amplitudes, cluster_rank, index) = context;
        let r = rank(reference, &det);
        if start > 0 {
            out.push(((index, det.alpha, det.beta), coefficient));
        }
        for (k, &(excitor, t)) in amplitudes.iter().enumerate().skip(start) {
            if r + rank(reference, &excitor.det) > cluster_rank {
                break;
            }
            if let Some((next, sign)) = excitor.apply(&det) {
                visit(next, -coefficient * t * sign, k + 1, context, out);
            }
        }
    }

    q.par_iter()
        .filter(|(_, value)| *value != 0.0)
        .fold(Vec::new, |mut out, &((alpha, beta), value)| {
            let nu = ExcitedDeterminant {
                parent: reference.parent,
                alpha,
                beta,
            };
            // The empty set enters once with weight one, the other sets with `P^{|K|-1}`.
            if empty && rank(reference, &nu) > 0 {
                out.push(((index, alpha, beta), value));
            }
            visit(
                nu,
                value / ranks,
                0,
                (reference, &amplitudes, cluster_rank, index),
                &mut out,
            );
            out
        })
        .flatten()
        .collect()
}
