// stochastic/overlapweighted.rs

// External crate imports.
use rand::Rng;

// Crate-root imports.
use crate::noci::{NOCIData, OverlapFactors, SpinFactorisation};

/// Outcome of one overlap-weighted branch proposal.
pub(in crate::stochastic) enum OverlapProposal {
    /// A real determinant was sampled and should be sent for batched matrix-element evaluation.
    Valid {
        /// Child determinant index.
        child: usize,
        /// Exact total mixture generation probability.
        pgen: f64,
    },
    /// The sampled Cartesian spin-component pair was invalid or had zero proposal mass.
    Null,
}

struct ParentDetLookup {
    /// Number of beta components in this parent lookup.
    nb: usize,
    /// Dense determinant table keyed by `a * nb + b`, with `usize::MAX` for absent pairs.
    det_by_ab: Vec<usize>,
}

struct DetOverlapMeta {
    /// Parent reference ID for this determinant.
    parent: usize,
    /// Parent-local alpha component ID for this determinant.
    a: usize,
    /// Parent-local beta component ID for this determinant.
    b: usize,
    /// Source determinant overlap-branch normalisation `Z_x`.
    z: f64,
}

/// Immutable overlap-weighted excitation generator data.
pub(in crate::stochastic) struct OverlapWeightedGenerator {
    /// Parent-local determinant lookup tables keyed by `(Q,a_w,b_w)`.
    det_by_ab: Vec<ParentDetLookup>,
    /// Per-determinant parent, component and overlap-normalisation metadata.
    det_meta: Vec<DetOverlapMeta>,
}

impl OverlapWeightedGenerator {
    /// Construct immutable lookup data for overlap-weighted excitation generation.
    /// # Arguments:
    /// - `data`: Shared NOCI data defining source determinant parents.
    /// - `spin`: Shared determinant-space spin factorisation.
    /// - `factors`: Persistent cross-parent overlap factors with CDF totals.
    /// # Returns:
    /// - `OverlapWeightedGenerator`: Immutable generator topology and configuration.
    pub(in crate::stochastic) fn new(
        data: &NOCIData<'_, f64>,
        spin: &SpinFactorisation,
        factors: &OverlapFactors,
    ) -> Self {
        let det_by_ab = (0..spin.nparents())
            .map(|parent| {
                let (na, nb) = spin.parent_component_counts(parent);
                let len = na
                    .checked_mul(nb)
                    .expect("parent spin-component lookup size overflowed usize");
                let mut det_by_ab = vec![usize::MAX; len];
                for entry in spin.parent_entries(parent) {
                    det_by_ab[entry.a * nb + entry.b] = entry.det;
                }
                ParentDetLookup { nb, det_by_ab }
            })
            .collect();

        let nparent = spin.nparents();
        let det_meta = (0..data.basis.len())
            .map(|det| {
                let source_parent = data.basis[det].parent;
                let source_a = spin.aid(det);
                let source_b = spin.bid(det);
                let mut ztotal = 0.0;

                for target_parent in 0..nparent {
                    if target_parent == source_parent {
                        continue;
                    }
                    if let Some(block) = factors.block(nparent, target_parent, source_parent) {
                        ztotal += block.alpha_total(source_a) * block.beta_total(source_b);
                    }
                }

                DetOverlapMeta {
                    parent: source_parent,
                    a: source_a,
                    b: source_b,
                    z: ztotal,
                }
            })
            .collect();

        Self {
            det_by_ab,
            det_meta,
        }
    }

    /// Sample one factorised-overlap branch proposal.
    /// If `(Q,a_w,b_w)` is not an actual determinant, returns a null proposal and does not redraw.
    /// # Arguments:
    /// - `self`: Overlap-weighted generator.
    /// - `source_det`: Source determinant `x`.
    /// - `factors`: Persistent cross-parent overlap factors with CDFs.
    /// - `overlap_weight`: Current report mixture probability `p`.
    /// - `rng`: Random-number generator.
    /// # Returns:
    /// - `OverlapProposal`: Valid determinant proposal or deliberate null proposal.
    pub(in crate::stochastic) fn sample_overlap<R: Rng + ?Sized>(
        &self,
        source_det: usize,
        factors: &OverlapFactors,
        overlap_weight: f64,
        rng: &mut R,
    ) -> OverlapProposal {
        let nparent = self.det_by_ab.len();
        let source = &self.det_meta[source_det];

        let ztotal = source.z;
        if ztotal == 0.0 {
            return OverlapProposal::Null;
        }

        let mut draw = rng.r#gen::<f64>() * ztotal;
        let mut target_parent = source.parent;
        let mut target_weight = 0.0;
        for parent in 0..nparent {
            if parent == source.parent {
                continue;
            }
            if let Some(block) = factors.block(nparent, parent, source.parent) {
                target_weight = block.alpha_total(source.a) * block.beta_total(source.b);
                if draw < target_weight {
                    target_parent = parent;
                    break;
                }
                draw -= target_weight;
            }
        }
        if target_parent == source.parent || target_weight == 0.0 {
            return OverlapProposal::Null;
        }

        let block = factors
            .block(nparent, target_parent, source.parent)
            .expect("sampled cross-parent overlap block must exist");
        let za = block.alpha_total(source.a);
        let zb = block.beta_total(source.b);
        if za == 0.0 || zb == 0.0 {
            return OverlapProposal::Null;
        }

        let target_a = block.sample_alpha(source.a, rng.r#gen::<f64>() * za);
        let target_b = block.sample_beta(source.b, rng.r#gen::<f64>() * zb);
        let Some(child) = self.lookup(target_parent, target_a, target_b) else {
            return OverlapProposal::Null;
        };

        // The factorised branch samples q_S(w|x) = |A^{QP}_{a_w a_x} B^{QP}_{b_w b_x}|/Z_x.
        // The accepted request carries the total mixture q_p = p q_S + (1 - p) q_U.
        let q_s = block.factor_abs(target_a, target_b, source.a, source.b) / ztotal;
        let q_u = 1.0 / (self.det_meta.len() - 1) as f64;
        let pgen = overlap_weight * q_s + (1.0 - overlap_weight) * q_u;

        OverlapProposal::Valid { child, pgen }
    }

    /// Evaluate the exact total mixture probability for an actual determinant pair.
    /// # Arguments:
    /// - `self`: Overlap-weighted generator.
    /// - `source_det`: Source determinant `x`.
    /// - `target_det`: Target determinant `w`.
    /// - `factors`: Persistent cross-parent overlap factors with CDFs.
    /// - `overlap_weight`: Current report mixture probability `p`.
    /// # Returns:
    /// - `f64`: Exact mixture generation probability `p q_S + (1 - p) q_U`.
    pub(in crate::stochastic) fn mixture_probability(
        &self,
        source_det: usize,
        target_det: usize,
        factors: &OverlapFactors,
        overlap_weight: f64,
    ) -> f64 {
        let q_u = 1.0 / (self.det_meta.len() - 1) as f64;
        let q_s = self.overlap_probability(source_det, target_det, factors);

        // Every accepted determinant is weighted by the full proposal
        // q_p(w|x) = p q_S(w|x) + (1 - p) q_U(w|x), independent of the branch that drew it.
        overlap_weight * q_s + (1.0 - overlap_weight) * q_u
    }

    /// Evaluate `q_S(w|x) = |S_{wx}|/Z_x` for an actual determinant pair.
    /// Same-parent and zero-overlap cross-parent determinant pairs return zero.
    /// # Arguments:
    /// - `self`: Overlap-weighted generator.
    /// - `source_det`: Source determinant `x`.
    /// - `target_det`: Target determinant `w`.
    /// - `factors`: Persistent cross-parent overlap factors with CDFs.
    /// # Returns:
    /// - `f64`: Factorised-overlap proposal probability `q_S(w|x)`.
    pub(in crate::stochastic) fn overlap_probability(
        &self,
        source_det: usize,
        target_det: usize,
        factors: &OverlapFactors,
    ) -> f64 {
        let nparent = self.det_by_ab.len();
        let source = &self.det_meta[source_det];
        let target = &self.det_meta[target_det];
        let mut q_s = 0.0;

        if target.parent != source.parent {
            let ztotal = source.z;
            if ztotal != 0.0
                && let Some(block) = factors.block(nparent, target.parent, source.parent)
            {
                // Same-parent and zero-overlap pairs have q_S(w|x) = 0. Cross-parent overlap
                // pairs use q_S(w|x) = |S_{wx}|/Z_x from cached A/B factors only.
                q_s = block.factor_abs(target.a, target.b, source.a, source.b) / ztotal;
            }
        }

        q_s
    }

    /// Return the determinant for one parent-local spin-component pair.
    /// # Arguments:
    /// - `self`: Overlap-weighted generator.
    /// - `parent`: Target parent `Q`.
    /// - `a`: Parent-local alpha component.
    /// - `b`: Parent-local beta component.
    /// # Returns:
    /// - `Option<usize>`: Determinant index when `(Q,a,b)` exists.
    fn lookup(
        &self,
        parent: usize,
        a: usize,
        b: usize,
    ) -> Option<usize> {
        let lookup = &self.det_by_ab[parent];
        let det = lookup.det_by_ab[a * lookup.nb + b];
        if det == usize::MAX { None } else { Some(det) }
    }
}
