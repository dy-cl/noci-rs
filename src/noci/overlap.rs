// noci/overlap.rs

use crate::DetState;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksPairView, WicksView};
use crate::nonorthogonalwicks::{lg_overlap, prepare_same};
use crate::time_call;

use super::naive::{build_s_pair, occ_coeffs};
use super::types::{DetPair, NOCIData, NOCIScalar};

/// Thread-local scratch for determinant to spin mappings.
pub(crate) struct OverlapFactorScratch {
    /// Cached alpha same-spin overlap values.
    avals: Vec<f64>,
    /// Epoch stamps for cached alpha values.
    astamps: Vec<u32>,
    /// Cached beta same-spin overlap values.
    bvals: Vec<f64>,
    /// Epoch stamps for cached beta values.
    bstamps: Vec<u32>,
    /// Current cache-validity epoch.
    epoch: u32,
}

/// Precomputed determinant to spin mappings for reusing alpha and beta overlap factors.
pub(crate) struct OverlapFactor {
    /// Alpha compact IDs keyed by determinant index and local to the determinant parent.
    aids: Vec<usize>,
    /// Beta compact IDs keyed by determinant index and local to the determinant parent.
    bids: Vec<usize>,
    /// Largest number of unique alpha spin components in one parent reference.
    ma: usize,
    /// Largest number of unique beta spin components in one parent reference.
    mb: usize,
}

impl OverlapFactor {
    /// Construct parent-local alpha and beta component IDs for every determinant.
    /// # Arguments:
    /// - `data`: Shared NOCI data defining the determinant basis and parent references.
    /// # Returns:
    /// - `OverlapAction`: Immutable sparse overlap action plan.
    pub(crate) fn new(data: &NOCIData<'_, f64>) -> Self {
        // Allocate determinant to ID mappings.
        let mut aids = vec![0usize; data.basis.len()];
        let mut bids = vec![0usize; data.basis.len()];
        // Generate determinant IDs.
        let ma = assign_aids(data, &mut aids);
        let mb = assign_bids(data, &mut bids);

        Self { aids, bids, ma, mb }
    }

    /// Construct parent-local alpha and beta component IDs for every determinant.
    /// # Arguments:
    /// - `self`: Immutable sparse overlap action plan.
    /// # Returns:
    /// - `OverlapActionScratch`: Thread-local sparse overlap scratch.
    pub(crate) fn scratch(&self) -> OverlapFactorScratch {
        let na = 2 * self.ma;
        let nb = 2 * self.mb;

        OverlapFactorScratch {
            avals: vec![0.0; na],
            astamps: vec![0; na],
            bvals: vec![0.0; nb],
            bstamps: vec![0; nb],
            epoch: 0,
        }
    }

    /// Evaluate one row of \delta p_\Gamma = \sum_\Omega S_{\Gamma\Omega}\delta n_\Omega from an ordered
    /// list of population changes. Reuse alpha and beta overlap factors shared by determinants with
    /// common spin components.
    /// # Arguments:
    /// - `gamma`: Determinant \Gamma defining the overlap row.
    /// - `updates`: Ordered pairs \((\Omega,\delta n_\Omega)\).
    /// - `data`: Shared NOCI data.
    /// - `wick_scratch`: Scratch space for Wick's calculations.
    /// - `scratch`: Thread-local reusable overlap factor scratch.
    /// # Returns:
    /// - `f64`: \delta p_\Gamma = \sum_\Omega S_{\Gamma\Omega}\delta n_\Omega.
    pub(crate) fn apply_row<I>(
        &self,
        gamma: usize,
        updates: I,
        data: &NOCIData<'_, f64>,
        wick_scratch: &mut WickScratchSpin<f64>,
        scratch: &mut OverlapFactorScratch,
    ) -> f64
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        let mut parent = usize::MAX;
        let mut dp = 0.0;

        // Prevent overflow of the epoch counter.
        if scratch.epoch == u32::MAX {
            // If we do overflow invalidate cache entries and restart counter.
            scratch.astamps.fill(0);
            scratch.bstamps.fill(0);
            scratch.epoch = 1;
        } else {
            // Otherwise increment as usual.
            scratch.epoch += 1;
        }

        for (omega, dn) in updates {
            // Early exit for a zero population change.
            if dn == 0.0 {
                continue;
            }

            // Alpha and Beta IDs `aids` and `bids` are local to one parent. When the parent
            // changes we must invalidate previous factors calculated with the previous parent.
            if data.basis[omega].parent != parent {
                // Prevent overflow of the epoch counter.
                if parent != usize::MAX {
                    if scratch.epoch == u32::MAX {
                        scratch.astamps.fill(0);
                        scratch.bstamps.fill(0);
                        scratch.epoch = 1;
                    } else {
                        scratch.epoch += 1;
                    }
                }

                // Record that parent has been updated.
                parent = data.basis[omega].parent;
            }

            // Order determinant pair consistently with overlap evaluators.
            let (ldet, gdet, left) = if gamma <= omega {
                (&data.basis[gamma], &data.basis[omega], true)
            } else {
                (&data.basis[omega], &data.basis[gamma], false)
            };

            // If \Gamma parent is the same as the current parent we can
            // use the fast orthogonal overlap matrix element path.
            if data.basis[gamma].parent == parent {
                dp += calculate_s_pair_orthogonal(ldet, gdet) * dn;
                continue;
            }

            // Otherwise we use the naive path if no Wicks.
            if !data.input.wicks.enabled {
                dp += calculate_s_pair_naive(data, ldet, gdet) * dn;
                continue;
            }

            // Retrieve the Wick intermediates used by the cached alpha and beta path.
            let Some(wicks) = data.wicks else {
                dp += calculate_s_pair_naive(data, ldet, gdet) * dn;
                continue;
            };

            // Same spin component can occur on either side of ordered determinant pair.
            // Therefore we keep two slots for each ID.
            let orient = usize::from(left);
            let aslot = 2 * self.aids[omega] + orient;
            let bslot = 2 * self.bids[omega] + orient;

            // Construct Wicks view if a spin factor not already cached.
            let mut pair: Option<WicksPairView<'_, f64>> = None;

            // If the stamp matches we have already evaluated this alpha
            // component for the current \Gamma.
            let sa = if scratch.astamps[aslot] == scratch.epoch {
                scratch.avals[aslot]
            } else {
                // Otherwise we need to calculate it.
                let w = get_pair(wicks, ldet.parent, gdet.parent, &mut pair);
                let sa = calculate_s_alpha_pair_wicks(ldet, gdet, &w, wick_scratch);

                // Cache the alpha factor for later determinants.
                scratch.avals[aslot] = sa;
                scratch.astamps[aslot] = scratch.epoch;
                sa
            };

            // Early exit for if the overlap matrix element is zero.
            if sa == 0.0 {
                continue;
            }

            // If the stamp matches we have already evaluated this beta
            // component for the current \Gamma.
            let sb = if scratch.bstamps[bslot] == scratch.epoch {
                scratch.bvals[bslot]
            } else {
                // Otherwise we need to calculate it.
                let w = get_pair(wicks, ldet.parent, gdet.parent, &mut pair);
                let sb = calculate_s_beta_pair_wicks(ldet, gdet, &w, wick_scratch);

                // Cache the beta factor for later determinants.
                scratch.bvals[bslot] = sb;
                scratch.bstamps[bslot] = scratch.epoch;
                sb
            };

            // Early exit for if the overlap matrix element is zero.
            if sb == 0.0 {
                continue;
            }

            dp += sa * sb * dn;
        }

        dp
    }
}

/// Assign compact alpha IDs by sorting determinant indices and deduplicating consecutive identities.
/// # Arguments:
/// - `data`: Shared NOCI data defining the determinant basis.
/// - `aids`: Output alpha compact IDs keyed by determinant index.
/// # Returns:
/// - `usize`: Largest number of unique alpha components in any parent.
fn assign_aids(
    data: &NOCIData<'_, f64>,
    aids: &mut [usize],
) -> usize {
    let mut indices = (0..data.basis.len()).collect::<Vec<_>>();

    // Sort by the alpha ID key such that equivalent determinants are consecutive.
    indices.sort_unstable_by(|&i, &j| {
        let id = &data.basis[i];
        let jd = &data.basis[j];

        // Sort by the ID key:
        // Parent reference, alpha occupation bitstring,
        // alpha holes, alpha particles, and alpha phase.
        id.parent
            .cmp(&jd.parent)
            .then_with(|| id.oa.cmp(&jd.oa))
            .then_with(|| id.excitation.alpha.holes.cmp(&jd.excitation.alpha.holes))
            .then_with(|| id.excitation.alpha.parts.cmp(&jd.excitation.alpha.parts))
            .then_with(|| id.pha.to_bits().cmp(&jd.pha.to_bits()))
    });

    // Previous determinant in sorted alpha key order.
    let mut last = usize::MAX;
    // Next unused alpha ID and number of unique components for a given parent.
    let mut next = 0usize;
    // Largest number of unique alpha components for any parent.
    let mut maxu = 0usize;

    // Enumerate sorted determinants.
    for (pos, &det) in indices.iter().enumerate() {
        let parent = data.basis[det].parent;
        // Detect first determinant or a new parent.
        if pos == 0 || parent != data.basis[last].parent {
            // If not the first determinant check if a new maximum number
            // of unique alpha components has been found for the previous parent.
            if pos != 0 {
                maxu = maxu.max(next);
            }
            // Reset IDs for the new parent.
            next = 0;
            aids[det] = next;
            next += 1;
        // Otherwise we have current determinant same parent as previous.
        } else {
            let prev = &data.basis[last];
            let curr = &data.basis[det];

            // If all keys match they get the same alpha ID.
            if prev.oa == curr.oa
                && prev.excitation.alpha.holes == curr.excitation.alpha.holes
                && prev.excitation.alpha.parts == curr.excitation.alpha.parts
                && prev.pha.to_bits() == curr.pha.to_bits()
            {
                aids[det] = aids[last];
            // Assign a new ID otherwise.
            } else {
                aids[det] = next;
                next += 1;
            }
        }
        last = det;
    }

    maxu.max(next)
}

/// Assign compact beta IDs by sorting determinant indices and deduplicating consecutive identities.
/// # Arguments:
/// - `data`: Shared NOCI data defining the determinant basis.
/// - `bids`: Output beta compact IDs keyed by determinant index.
/// # Returns:
/// - `usize`: Largest number of unique beta components in any parent.
fn assign_bids(
    data: &NOCIData<'_, f64>,
    bids: &mut [usize],
) -> usize {
    let mut indices = (0..data.basis.len()).collect::<Vec<_>>();

    // Sort by the beta ID key such that equivalent determinants are consecutive.
    indices.sort_unstable_by(|&i, &j| {
        let id = &data.basis[i];
        let jd = &data.basis[j];

        // Sort by the ID key:
        // Parent reference, beta occupation bitstring,
        // beta holes, beta particles, and beta phase.
        id.parent
            .cmp(&jd.parent)
            .then_with(|| id.ob.cmp(&jd.ob))
            .then_with(|| id.excitation.beta.holes.cmp(&jd.excitation.beta.holes))
            .then_with(|| id.excitation.beta.parts.cmp(&jd.excitation.beta.parts))
            .then_with(|| id.phb.to_bits().cmp(&jd.phb.to_bits()))
    });

    // Previous determinant in sorted beta key order.
    let mut last = usize::MAX;
    // Next unused beta ID and number of unique components for a given parent.
    let mut next = 0usize;
    // Largest number of unique beta components for any parent.
    let mut maxu = 0usize;

    // Enumerate sorted determinants.
    for (pos, &det) in indices.iter().enumerate() {
        let parent = data.basis[det].parent;

        // Detect first determinant or a new parent.
        if pos == 0 || parent != data.basis[last].parent {
            // If not the first determinant check if a new maximum number
            // of unique beta components has been found for the previous parent.
            if pos != 0 {
                maxu = maxu.max(next);
            }

            // Reset IDs for the new parent.
            next = 0;
            bids[det] = next;
            next += 1;
        // Otherwise we have current determinant same parent as previous.
        } else {
            let prev = &data.basis[last];
            let curr = &data.basis[det];

            // If all keys match they get the same beta ID.
            if prev.ob == curr.ob
                && prev.excitation.beta.holes == curr.excitation.beta.holes
                && prev.excitation.beta.parts == curr.excitation.beta.parts
                && prev.phb.to_bits() == curr.phb.to_bits()
            {
                bids[det] = bids[last];
            // Assign a new ID otherwise.
            } else {
                bids[det] = next;
                next += 1;
            }
        }

        last = det;
    }

    maxu.max(next)
}

/// Return a Wick pair view, constructing it at most once for the current update.
/// # Arguments:
/// - `wicks`: Wick intermediates for all parent pairs.
/// - `lp`: Left parent reference index.
/// - `gp`: Right parent reference index.
/// - `pair`: Optional cached pair view for the current update.
/// # Returns:
/// - `WicksPairView<'_, f64>`: Wick pair view for the ordered parent pair.
#[inline(always)]
fn get_pair<'a>(
    wicks: &'a WicksView<f64>,
    lp: usize,
    gp: usize,
    pair: &mut Option<WicksPairView<'a, f64>>,
) -> WicksPairView<'a, f64> {
    if let Some(w) = *pair {
        w
    } else {
        wicks.prefetch_pair(lp, gp);
        let w = wicks.pair(lp, gp);
        *pair = Some(w);
        w
    }
}

/// Wrapper function which dispatches to overlap matrix-element evaluation routines depending on
/// user input and properties of the determinant pair involved.
/// If the determinant pair have the same Hermitian-orthonormal parent we may use the standard
/// Slater-Condon rules, if not we can either use generalised Slater-Condon rules or extended
/// non-orthogonal Wick's theorem to evaluate the matrix element.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose overlap matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element between the determinant pair.
pub(crate) fn calculate_s_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair, {
        let ldet = pair.ldet;
        let gdet = pair.gdet;

        if ldet.parent == gdet.parent {
            let mocache = data
                .mocache
                .expect("Orthogonal overlap matrix elements require mocache.");
            if mocache[ldet.parent].orthogonal_slater_condon {
                return calculate_s_pair_orthogonal(ldet, gdet);
            }
        }

        if data.input.wicks.enabled {
            calculate_s_pair_wicks(ldet, gdet, data.wicks.unwrap(), scratch.unwrap())
        } else {
            calculate_s_pair_naive(data, ldet, gdet)
        }
    })
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `T`: Overlap matrix element between `ldet` and `gdet`.
pub(in crate::noci) fn calculate_s_pair_orthogonal<T: NOCIScalar>(
    ldet: &DetState<T>,
    gdet: &DetState<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_orthogonal, {
        if ldet.oa == gdet.oa && ldet.ob == gdet.ob {
            <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb))
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using
/// generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `T`: Overlap matrix element between `ldet` and `gdet`.
fn calculate_s_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_naive, {
        let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
        let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
        let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
        let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

        let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
        det_phase * pa.s * pb.s
    })
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma
/// using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element.
fn calculate_s_pair_wicks<T: NOCIScalar>(
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_wicks, {
        let w = wicks.pair(ldet.parent, gdet.parent);

        let ex_la = &ldet.excitation.alpha;
        let ex_ga = &gdet.excitation.alpha;
        let ex_lb = &ldet.excitation.beta;
        let ex_gb = &gdet.excitation.beta;

        let la = ex_la.holes.len() + ex_ga.holes.len();
        let lb = ex_lb.holes.len() + ex_gb.holes.len();

        if w.aa.m > la || w.bb.m > lb {
            return <T as From<f64>>::from(0.0);
        }

        let pha = <T as From<f64>>::from(ldet.pha * gdet.pha);
        let phb = <T as From<f64>>::from(ldet.phb * gdet.phb);
        let zero = <T as From<f64>>::from(0.0);

        prepare_same(&w.aa, ex_la, ex_ga, &mut scratch.aa);
        let sa = pha * lg_overlap(&w.aa, ex_la, ex_ga, &mut scratch.aa);

        if sa == zero {
            return zero;
        }

        prepare_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
        let sb = phb * lg_overlap(&w.bb, ex_lb, ex_gb, &mut scratch.bb);

        if sb == zero {
            return zero;
        }

        sa * sb
    })
}

/// Calculate the alpha same-spin overlap for an ordered Wick pair.
/// # Arguments:
/// - `ldet`: Left determinant.
/// - `gdet`: Right determinant.
/// - `w`: Wick intermediates for the ordered parent pair.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Alpha same-spin overlap including determinant phases.
#[inline(always)]
pub(in crate::noci) fn calculate_s_alpha_pair_wicks(
    ldet: &DetState<f64>,
    gdet: &DetState<f64>,
    w: &WicksPairView<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    let l_ex = &ldet.excitation.alpha;
    let g_ex = &gdet.excitation.alpha;
    let l = l_ex.holes.len() + g_ex.holes.len();

    if w.aa.m > l {
        return 0.0;
    }

    prepare_same(&w.aa, l_ex, g_ex, &mut scratch.aa);
    ldet.pha * gdet.pha * lg_overlap(&w.aa, l_ex, g_ex, &mut scratch.aa)
}

/// Calculate the beta same-spin overlap for an ordered Wick pair.
/// # Arguments:
/// - `ldet`: Left determinant.
/// - `gdet`: Right determinant.
/// - `w`: Wick intermediates for the ordered parent pair.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Beta same-spin overlap including determinant phases.
#[inline(always)]
pub(in crate::noci) fn calculate_s_beta_pair_wicks(
    ldet: &DetState<f64>,
    gdet: &DetState<f64>,
    w: &WicksPairView<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    let l_ex = &ldet.excitation.beta;
    let g_ex = &gdet.excitation.beta;
    let l = l_ex.holes.len() + g_ex.holes.len();

    if w.bb.m > l {
        return 0.0;
    }

    prepare_same(&w.bb, l_ex, g_ex, &mut scratch.bb);
    ldet.phb * gdet.phb * lg_overlap(&w.bb, l_ex, g_ex, &mut scratch.bb)
}
