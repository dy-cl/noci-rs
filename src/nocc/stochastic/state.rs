// nocc/stochastic/state.rs

// External crate imports.
use rand::SeedableRng;

// Crate-root imports.
use crate::determinant::DeterminantState;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{DetPair, HamiltonianOverlap, NOCIData, Overlap, calculate_pairs};
use crate::hash::FastMap;
use crate::qmc::{QmcRng, Worker};

// Parent/sibling imports.
use super::coupling::Projections;
use super::population::{Excitor, ExcitorIndex, ReferenceSpaces};

/// Destination of one batched same-reference Hamiltonian term.
#[derive(Clone, Copy)]
pub(super) enum Target {
    /// Contribution to `N_zH^e_{zz}` of block `z`.
    Energy(usize),
    /// Population change per unit time of one excitor.
    Excitor(ExcitorIndex),
}

/// Batched same-reference request: `coefficient\,\langle\text{bra}|\hat H|\text{ket}\rangle` is
/// added to `target`.
pub(super) struct LinkedBatchedRequest {
    /// Destination of the term.
    pub(super) target: Target,
    /// Weight, sign and inverse proposal probability multiplying the element.
    pub(super) coefficient: f64,
    /// Bra determinant.
    pub(super) bra: DeterminantState,
    /// Ket determinant.
    pub(super) ket: DeterminantState,
}

/// Batched coupling request projecting one sampled cluster of block `x` onto one row of block `y`:
/// the reference row gives `\langle{}^y\Psi|D_C\rangle` and `\langle{}^y\Psi|\hat H|D_C\rangle`,
/// any other row `\nu` gives `\langle D_\nu|D_C\rangle`, each multiplied by `coefficient`.
pub(super) struct CouplingBatchedRequest {
    /// Index of the prepared cluster determinant in the worker's cluster list.
    pub(super) cluster: usize,
    /// Source block `x`.
    pub(super) x: usize,
    /// Projection block `y`.
    pub(super) y: usize,
    /// Projection row of block `y`, or `None` for its reference.
    pub(super) row: Option<usize>,
    /// Cluster weight, scaled by the inverse row-sampling probability for non-reference rows.
    pub(super) coefficient: f64,
}

/// Persistent per-thread worker of cluster attempts. Matrix-element requests are queued during a
/// cycle's attempts and resolved together, as in the NOCIQMC propagation workers.
pub(super) struct JMThreadPropagation {
    /// Population changes per unit time, summed per excitor.
    pub(super) spawns: FastMap<ExcitorIndex, f64>,
    /// Contribution to `N_z H^e_{zz}` of each block.
    pub(super) energy: Vec<f64>,
    /// Coupling projections.
    pub(super) projections: Projections,
    /// Excitors of the current cluster.
    pub(super) excitors: Vec<Excitor>,
    /// Queued same-reference requests.
    pub(super) linked_requests: Vec<LinkedBatchedRequest>,
    /// Prepared cluster determinants of the queued coupling requests.
    pub(super) coupling_clusters: Vec<DeterminantState>,
    /// Queued coupling requests.
    pub(super) coupling_requests: Vec<CouplingBatchedRequest>,
    /// Determinant pairs of the queued same-reference requests.
    linked_pairs: Vec<DetPair>,
    /// Hamiltonian and overlap elements aligned with `linked_pairs`.
    linked_hs: Vec<(f64, f64)>,
    /// Hamiltonian and overlap elements of the reference-row coupling requests.
    coupling_hs: Vec<(f64, f64)>,
    /// Overlap elements of the other coupling requests.
    coupling_s: Vec<f64>,
    /// Scratch space for the extended nonorthogonal Wick theorem.
    wick_scratch: Box<WickScratchSpin<f64>>,
    /// Random stream of this worker's thread and cycle.
    pub(super) rng: QmcRng,
}

impl JMThreadPropagation {
    /// Construct an empty worker for `M` references.
    /// # Arguments:
    /// - `m`: Number of references.
    /// - `seed`: Seed of the worker's stream before its first cycle.
    /// # Returns:
    /// - `JMThreadPropagation`: Empty worker.
    pub(super) fn new(
        m: usize,
        seed: u64,
    ) -> Self {
        Self {
            spawns: FastMap::default(),
            energy: vec![0.0; m],
            projections: Projections::new(m),
            excitors: Vec::new(),
            linked_requests: Vec::new(),
            coupling_clusters: Vec::new(),
            coupling_requests: Vec::new(),
            linked_pairs: Vec::new(),
            linked_hs: Vec::new(),
            coupling_hs: Vec::new(),
            coupling_s: Vec::new(),
            wick_scratch: Box::new(WickScratchSpin::new()),
            rng: QmcRng::seed_from_u64(seed),
        }
    }

    /// Clear the results of the previous cycle while retaining every allocation.
    /// # Arguments:
    /// - `self`: Worker.
    /// # Returns:
    /// - `()`: Clears the storage in place.
    pub(super) fn clear(&mut self) {
        self.spawns.clear();
        self.energy.iter_mut().for_each(|e| *e = 0.0);
        self.projections.clear();
        self.linked_requests.clear();
        self.coupling_clusters.clear();
        self.coupling_requests.clear();
    }

    /// Resolve queues that reached the request watermark after one complete cluster attempt.
    /// The 4096-request watermark is a throughput/memory tuning choice: it limits queued work
    /// while retaining enough pairs for batching. The H10 timings reported in PR #35 establish
    /// comparable throughput to the previous scheduler, rather than a cache capacity or a
    /// universal optimum. Changing this value can change floating-point grouping and seeded
    /// trajectories. One attempt may overshoot the watermark; no cluster is split between batches.
    /// # Arguments:
    /// - `self`: Worker holding the pending same-reference and coupling requests.
    /// - `data`: NOCI data holding integral caches and Wick intermediates.
    /// - `spaces`: Prepared reference and projection determinants.
    /// # Returns:
    /// - `()`: Accumulates any flushed requests and retains their queue allocations.
    pub(super) fn resolve_if_full(
        &mut self,
        data: &NOCIData<'_, f64>,
        spaces: &ReferenceSpaces,
    ) {
        const REQUESTLIMIT: usize = 4096;

        if self.linked_requests.len() >= REQUESTLIMIT {
            self.resolve_linked(data);
        }

        if self.coupling_requests.len() >= REQUESTLIMIT {
            self.resolve_coupling(data, spaces);
        }
    }

    /// Evaluate the queued same-reference requests in one batch and add each term to its target.
    /// # Arguments:
    /// - `self`: Worker.
    /// - `data`: NOCI data holding the parent frames and MO caches.
    /// # Returns:
    /// - `()`: Adds the terms to `spawns` and `energy` and clears the queue.
    pub(super) fn resolve_linked(
        &mut self,
        data: &NOCIData<'_, f64>,
    ) {
        if self.linked_requests.is_empty() {
            return;
        }

        // Materialise one pair batch in the same order as its signed target contributions.
        self.linked_pairs.clear();
        self.linked_pairs.extend(
            self.linked_requests
                .iter()
                .map(|r| DetPair::new(r.bra, r.ket)),
        );
        self.linked_hs.clear();
        self.linked_hs.resize(self.linked_pairs.len(), (0.0, 0.0));

        calculate_pairs(
            data,
            &HamiltonianOverlap,
            &self.linked_pairs,
            Some(&mut self.wick_scratch),
            &mut self.linked_hs,
        );

        // Scatter coefficient * H into the block energy or excitor residual.
        for (request, &(h, _)) in self.linked_requests.iter().zip(&self.linked_hs) {
            let value = request.coefficient * h;
            if value == 0.0 {
                continue;
            }
            match request.target {
                Target::Energy(z) => self.energy[z] += value,
                Target::Excitor(key) => *self.spawns.entry(key).or_insert(0.0) += value,
            }
        }

        self.linked_requests.clear();
    }

    /// Evaluate the queued coupling requests in one batch and add them to the projections.
    /// # Arguments:
    /// - `self`: Worker.
    /// - `data`: NOCI data holding the Wick intermediates.
    /// - `spaces`: Prepared references and projection rows of every block.
    /// # Returns:
    /// - `()`: Adds the projections and clears the queue.
    pub(super) fn resolve_coupling(
        &mut self,
        data: &NOCIData<'_, f64>,
        spaces: &ReferenceSpaces,
    ) {
        if self.coupling_requests.is_empty() {
            return;
        }

        // Reference projections need (H,S); excited projection rows need only S.
        let clusters = &self.coupling_clusters;
        let mut hs_pairs = Vec::new();
        let mut s_pairs = Vec::new();

        for request in &self.coupling_requests {
            let ket = clusters[request.cluster];
            match request.row {
                None => hs_pairs.push(DetPair::new(spaces.references[request.y], ket)),
                Some(r) => s_pairs.push(DetPair::new(spaces.rows[request.y][r], ket)),
            }
        }

        // Evaluate the two operator batches without changing request or accumulation order.
        self.coupling_hs.clear();
        self.coupling_hs.resize(hs_pairs.len(), (0.0, 0.0));
        calculate_pairs(
            data,
            &HamiltonianOverlap,
            &hs_pairs,
            Some(&mut self.wick_scratch),
            &mut self.coupling_hs,
        );
        self.coupling_s.clear();
        self.coupling_s.resize(s_pairs.len(), 0.0);
        calculate_pairs(
            data,
            &Overlap,
            &s_pairs,
            Some(&mut self.wick_scratch),
            &mut self.coupling_s,
        );

        // Restore the original projection order from the separate (H,S) and S batches.
        let (mut ihs, mut is) = (0, 0);
        let out = &mut self.projections;
        for request in &self.coupling_requests {
            let (y, x, w) = (request.y, request.x, request.coefficient);
            match request.row {
                None => {
                    let (h, s) = self.coupling_hs[ihs];
                    ihs += 1;
                    let reference = &spaces.references[y];
                    out.se[y][x] += w * s;
                    out.he[y][x] += w * h;
                    *out.q[y][x]
                        .entry((reference.alpha, reference.beta))
                        .or_insert(0.0) += w * s;
                }
                Some(r) => {
                    let s = self.coupling_s[is];
                    is += 1;
                    if s != 0.0 {
                        let nu = &spaces.rows[y][r];
                        *out.q[y][x].entry((nu.alpha, nu.beta)).or_insert(0.0) += w * s;
                    }
                }
            }
        }

        self.coupling_clusters.clear();
        self.coupling_requests.clear();
    }
}

impl Worker for JMThreadPropagation {
    /// Clear the worker's per-cycle buffers and give it the cycle's random stream.
    /// # Arguments:
    /// - `self`: JM-NOCCMC worker.
    /// - `rng`: Random stream of this worker's thread and the cycle.
    /// # Returns:
    /// - `()`: Resets the worker.
    fn reset(
        &mut self,
        rng: QmcRng,
    ) {
        self.clear();
        self.rng = rng;
    }
}
