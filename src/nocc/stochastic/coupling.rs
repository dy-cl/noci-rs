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

// Crate-root imports.
use crate::determinant::ExcitedDeterminant;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{NOCIData, calculate_hs_excited, calculate_s_excited};
use crate::qmc::QmcRng;

// Parent/sibling imports.
use super::population::{Block, Excitor, ExcitorKey, FixedMap, rank};

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
    pub(super) q: Vec<Vec<FixedMap<(u128, u128), f64>>>,
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
                .map(|_| (0..m).map(|_| FixedMap::default()).collect())
                .collect(),
        }
    }

    /// Add another set of projections.
    /// # Arguments:
    /// - `self`: Accumulated projections.
    /// - `other`: Projections to add.
    /// # Returns:
    /// - `()`: Adds `other` in place.
    pub(super) fn add(
        &mut self,
        other: &Projections,
    ) {
        for (y, row) in other.q.iter().enumerate() {
            for (x, map) in row.iter().enumerate() {
                self.se[y][x] += other.se[y][x];
                self.he[y][x] += other.he[y][x];
                for (&nu, &v) in map {
                    *self.q[y][x].entry(nu).or_insert(0.0) += v;
                }
            }
        }
    }
}

/// Project one sampled cluster `w\,\phi_C|D_C\rangle` of block `x` onto every other block `y`:
/// the reference row exactly, giving `\langle{}^y\Psi|D_C\rangle` and
/// `\langle{}^y\Psi|\hat H|D_C\rangle`, and `ROWDRAWS` uniformly drawn rows `\nu` of rank
/// `1..=cluster_rank` with weight `n_{\text{rows}}/\text{ROWDRAWS}`.
/// # Arguments:
/// - `data`: NOCI data holding the parent frames and Wick intermediates.
/// - `blocks`: Block populations.
/// - `x`: Source block.
/// - `cluster`: Collapsed determinant `|D_C\rangle` and signed weight `w\phi_C`.
/// - `rng`: Random-number generator.
/// - `scratch`: Wick scratch.
/// - `out`: Accumulated projections.
/// # Returns:
/// - `()`: Adds the cluster's projections.
pub(super) fn project_cluster(
    data: &NOCIData<'_, f64>,
    blocks: &[Block],
    x: usize,
    cluster: (&ExcitedDeterminant, f64),
    rng: &mut QmcRng,
    scratch: &mut WickScratchSpin<f64>,
    out: &mut Projections,
) {
    let (det, weight) = cluster;
    for (y, block) in blocks.iter().enumerate() {
        if y == x {
            continue;
        }

        // Reference row: effective overlap and Hamiltonian.
        let (h, s) = calculate_hs_excited(data, (&block.reference, det), scratch);
        out.se[y][x] += weight * s;
        out.he[y][x] += weight * h;
        let reference = (block.reference.alpha, block.reference.beta);
        *out.q[y][x].entry(reference).or_insert(0.0) += weight * s;

        // Remaining rows drawn uniformly.
        if block.rows.is_empty() {
            continue;
        }
        let scale = weight * block.rows.len() as f64 / ROWDRAWS as f64;
        for _ in 0..ROWDRAWS {
            let nu = &block.rows[rng.gen_range(0..block.rows.len())];
            let s = calculate_s_excited(data, (nu, det), scratch);
            if s != 0.0 {
                *out.q[y][x].entry((nu.alpha, nu.beta)).or_insert(0.0) += scale * s;
            }
        }
    }
}

/// Apply `e^{-\hat T_y}` from the left to a projection, giving for every projectee
/// `\mu \ne {}^y\Psi` of rank at most the cluster rank
/// `\sum_\nu\langle D_\mu|e^{-\hat T_y}|D_\nu\rangle q(\nu)` with
/// `\langle D_\mu|e^{-\hat T_y}|D_\nu\rangle = \sum_K(-1)^{|K|}\prod_{k\in K}t_{yk}\langle D_\mu|\hat\tau_K|D_\nu\rangle`
/// over sets `K` of distinct excitors, since a repeated excitor vanishes and the `|K|!` orderings
/// cancel the factorial of the exponential.
/// # Arguments:
/// - `block`: Population of block `y` with index `index`.
/// - `index`: Block index `y`.
/// - `q`: Combined projection `q(\nu)` keyed by occupations.
/// - `cluster_rank`: Highest excitation rank of the cluster operator.
/// - `scale`: Factor applied to every change, such as `\Delta\tau`.
/// - `out`: Population changes.
/// # Returns:
/// - `()`: Appends the changes.
pub(super) fn left_action(
    block: &Block,
    index: usize,
    q: &FixedMap<(u128, u128), f64>,
    cluster_rank: usize,
    scale: f64,
    out: &mut Vec<(ExcitorKey, f64)>,
) {
    let reference = &block.reference;
    let amplitudes = block
        .excitors
        .iter()
        .map(|&(excitor, n)| (excitor, n / block.nref))
        .collect::<Vec<(Excitor, f64)>>();

    // Depth-first enumeration of sets of distinct excitors in increasing order.
    fn visit(
        det: ExcitedDeterminant,
        coefficient: f64,
        start: usize,
        context: (&ExcitedDeterminant, &[(Excitor, f64)], usize, usize),
        out: &mut Vec<(ExcitorKey, f64)>,
    ) {
        let (reference, amplitudes, cluster_rank, index) = context;
        let r = rank(reference, &det);
        if r > 0 {
            out.push(((index, det.alpha, det.beta), coefficient));
        }
        for (k, &(excitor, t)) in amplitudes.iter().enumerate().skip(start) {
            if r + rank(reference, &excitor.det) > cluster_rank {
                continue;
            }
            if let Some((next, sign)) = excitor.apply(&det) {
                visit(next, -coefficient * t * sign, k + 1, context, out);
            }
        }
    }

    for (&(alpha, beta), &value) in q {
        if value == 0.0 {
            continue;
        }
        let nu = ExcitedDeterminant {
            parent: reference.parent,
            alpha,
            beta,
        };
        visit(
            nu,
            scale * value,
            0,
            (reference, &amplitudes, cluster_rank, index),
            out,
        );
    }
}
