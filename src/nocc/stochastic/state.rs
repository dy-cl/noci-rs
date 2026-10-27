// nocc/stochastic/state.rs

// Crate-root imports.
use crate::determinant::{ExcitedDeterminant, ExcitedState};
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{NOCIData, calculate_hs_excited, calculate_hs_states, calculate_s_states};
use crate::qmc::FastMap;

// Parent/sibling imports.
use super::coupling::Projections;
use super::population::{Excitor, ExcitorKey, ReferenceSpaces};

/// Destination of one batched same-reference Hamiltonian term.
#[derive(Clone, Copy)]
pub(super) enum Target {
    /// Contribution to `N_zH^e_{zz}` of block `z`.
    Energy(usize),
    /// Population change per unit time of one excitor.
    Excitor(ExcitorKey),
}

/// Batched same-reference request: `coefficient\,\langle\text{bra}|\hat H|\text{ket}\rangle` is
/// added to `target`.
pub(super) struct LinkedBatchedRequest {
    /// Destination of the term.
    pub(super) target: Target,
    /// Weight, sign and inverse proposal probability multiplying the element.
    pub(super) coefficient: f64,
    /// Bra determinant.
    pub(super) bra: ExcitedDeterminant,
    /// Ket determinant.
    pub(super) ket: ExcitedDeterminant,
}

/// Batched coupling request projecting one sampled cluster of block `x` onto one row of block `y`:
/// the reference row gives `\langle{}^y\Psi|D_C\rangle` and `\langle{}^y\Psi|\hat H|D_C\rangle`,
/// any other row `\nu` gives `\langle D_\nu|D_C\rangle`, each multiplied by `coefficient`.
pub(super) struct CouplingBatchedRequest {
    /// Index of the prepared cluster determinant in the task's cluster list.
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

/// Reusable storage of one parallel task of cluster attempts. Matrix-element requests are
/// queued during the attempts and resolved together, as in the NOCIQMC propagation workers.
pub(super) struct JMTaskPropagation {
    /// Population changes per unit time, summed per excitor.
    pub(super) spawns: FastMap<ExcitorKey, f64>,
    /// Contribution to `N_z H^e_{zz}` of each block.
    pub(super) energy: Vec<f64>,
    /// Coupling projections.
    pub(super) projections: Projections,
    /// Excitors of the current cluster.
    pub(super) excitors: Vec<Excitor>,
    /// Queued same-reference requests.
    pub(super) linked_requests: Vec<LinkedBatchedRequest>,
    /// Prepared cluster determinants of the queued coupling requests.
    pub(super) coupling_clusters: Vec<ExcitedState>,
    /// Queued coupling requests.
    pub(super) coupling_requests: Vec<CouplingBatchedRequest>,
    /// Determinant pairs of the queued same-reference requests.
    linked_pairs: Vec<(ExcitedDeterminant, ExcitedDeterminant)>,
    /// Hamiltonian and overlap elements aligned with `linked_pairs`.
    linked_hs: Vec<(f64, f64)>,
    /// Hamiltonian and overlap elements of the reference-row coupling requests.
    coupling_hs: Vec<(f64, f64)>,
    /// Overlap elements of the other coupling requests.
    coupling_s: Vec<f64>,
    /// Scratch space for the extended nonorthogonal Wick theorem.
    wick_scratch: Box<WickScratchSpin<f64>>,
}

impl JMTaskPropagation {
    /// Construct empty task storage for `M` references.
    /// # Arguments:
    /// - `m`: Number of references.
    /// # Returns:
    /// - `JMTaskPropagation`: Empty storage.
    pub(super) fn new(m: usize) -> Self {
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
        }
    }

    /// Clear the results of the previous task while retaining every allocation.
    /// # Arguments:
    /// - `self`: Task storage.
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

    /// Evaluate the queued same-reference requests in one batch and add each term to its target.
    /// # Arguments:
    /// - `self`: Task storage.
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
        self.linked_pairs.clear();
        self.linked_pairs
            .extend(self.linked_requests.iter().map(|r| (r.bra, r.ket)));
        self.linked_hs.clear();
        self.linked_hs.resize(self.linked_pairs.len(), (0.0, 0.0));
        calculate_hs_excited(
            data,
            &self.linked_pairs,
            &mut self.wick_scratch,
            &mut self.linked_hs,
        );

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
    /// - `self`: Task storage.
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
        let clusters = &self.coupling_clusters;
        let mut hs_pairs = Vec::new();
        let mut s_pairs = Vec::new();
        for request in &self.coupling_requests {
            let ket = &clusters[request.cluster];
            match request.row {
                None => hs_pairs.push((&spaces.states[request.y], ket)),
                Some(r) => s_pairs.push((&spaces.rows[request.y][r], ket)),
            }
        }
        self.coupling_hs.clear();
        self.coupling_hs.resize(hs_pairs.len(), (0.0, 0.0));
        calculate_hs_states(
            data,
            &hs_pairs,
            &mut self.wick_scratch,
            &mut self.coupling_hs,
        );
        self.coupling_s.clear();
        self.coupling_s.resize(s_pairs.len(), 0.0);
        calculate_s_states(data, &s_pairs, &mut self.wick_scratch, &mut self.coupling_s);

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
                        let nu = &spaces.rows[y][r].det;
                        *out.q[y][x].entry((nu.alpha, nu.beta)).or_insert(0.0) += w * s;
                    }
                }
            }
        }
        self.coupling_clusters.clear();
        self.coupling_requests.clear();
    }
}
