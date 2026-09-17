// stochastic/excit.rs
// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::Rng;

// Crate-root imports.
use crate::input::{ExcitationGen, Input};
use crate::noci::{MOCache, NOCIData, OrthogonalConnection};
use crate::nonorthogonalwicks::WickScratchSpin;

// Parent/sibling imports.
use super::common::find_hs;
use super::state::{HeatBath, OverlapDerivativeSums, PropagationState, QMCRunInfo, QmcRng};

/// Persistent uniform proposal topology for parent-orthogonal Hamiltonian connections.
pub(in crate::stochastic) struct OrthogonalUniformGenerator {
    /// Exact probability `1/N_\mathrm{conn}` for every table entry.
    pgen: f64,
    /// Complete system-wide table of relative occupied/virtual-rank connections.
    connections: Vec<OrthogonalConnection>,
}

impl OrthogonalUniformGenerator {
    /// Construct all `N_\mathrm{conn}` one- and two-body connection topologies once.
    /// The table enumerates `O_\alpha V_\alpha`, `O_\beta V_\beta`, same-spin pair products,
    /// and `O_\alpha V_\alpha O_\beta V_\beta` using orbital ranks rather than labels.
    /// # Arguments:
    /// - `occupations`: Representative alpha and beta occupations defining fixed electron counts.
    /// - `cache`: Representative MO cache defining common alpha and beta orbital dimensions.
    /// # Returns
    /// - `Self`: System-wide flat uniform connection table and valid-orbital masks.
    pub(in crate::stochastic) fn new(
        occupations: (u128, u128),
        cache: &MOCache<f64>,
    ) -> Self {
        // Derive occupied/virtual dimensions shared by every parent-orthogonal determinant.
        let noa = occupations.0.count_ones() as usize;
        let nob = occupations.1.count_ones() as usize;
        let nva = cache.ha.nrows() - noa;
        let nvb = cache.hb.nrows() - nob;
        // Count each Slater-Condon connection class and reserve the exact combined capacity.
        let nas = noa * nva;
        let nbs = nob * nvb;
        let naa = (noa * noa.saturating_sub(1) / 2) * (nva * nva.saturating_sub(1) / 2);
        let nbb = (nob * nob.saturating_sub(1) / 2) * (nvb * nvb.saturating_sub(1) / 2);
        let nab = nas * nbs;
        let mut connections = Vec::with_capacity(nas + nbs + naa + nbb + nab);

        // Enumerate alpha and beta single excitations in occupied/virtual rank coordinates.
        for occupied in 0..noa {
            for virtual_ in 0..nva {
                connections.push(OrthogonalConnection::AlphaSingle {
                    occupied: occupied as u8,
                    virtual_: virtual_ as u8,
                });
            }
        }
        for occupied in 0..nob {
            for virtual_ in 0..nvb {
                connections.push(OrthogonalConnection::BetaSingle {
                    occupied: occupied as u8,
                    virtual_: virtual_ as u8,
                });
            }
        }
        // Enumerate unique same-spin doubles with ordered occupied and virtual pairs.
        for occupied_i in 0..noa {
            for occupied_j in occupied_i + 1..noa {
                for virtual_a in 0..nva {
                    for virtual_b in virtual_a + 1..nva {
                        connections.push(OrthogonalConnection::AlphaDouble {
                            occupied: [occupied_i as u8, occupied_j as u8],
                            virtual_: [virtual_a as u8, virtual_b as u8],
                        });
                    }
                }
            }
        }
        for occupied_i in 0..nob {
            for occupied_j in occupied_i + 1..nob {
                for virtual_a in 0..nvb {
                    for virtual_b in virtual_a + 1..nvb {
                        connections.push(OrthogonalConnection::BetaDouble {
                            occupied: [occupied_i as u8, occupied_j as u8],
                            virtual_: [virtual_a as u8, virtual_b as u8],
                        });
                    }
                }
            }
        }
        // Enumerate all Cartesian products of alpha and beta single excitations.
        for occupied_a in 0..noa {
            for virtual_a in 0..nva {
                for occupied_b in 0..nob {
                    for virtual_b in 0..nvb {
                        connections.push(OrthogonalConnection::AlphaBetaDouble {
                            occupied_a: occupied_a as u8,
                            virtual_a: virtual_a as u8,
                            occupied_b: occupied_b as u8,
                            virtual_b: virtual_b as u8,
                        });
                    }
                }
            }
        }

        // Every retained topology has the same exact generation probability.
        let pgen = if connections.is_empty() {
            0.0
        } else {
            1.0 / connections.len() as f64
        };

        Self { pgen, connections }
    }

    /// Sample one connection with `P_\mathrm{gen}(D|x)=1/N_\mathrm{conn}`.
    /// # Arguments:
    /// - `self`: Persistent uniform connection topology.
    /// - `rng`: Thread-local random-number generator.
    /// # Returns
    /// - `Option<(usize, f64)>`: Connection-table index and exact uniform probability.
    #[inline(always)]
    pub(in crate::stochastic) fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Option<(usize, f64)> {
        if self.connections.is_empty() {
            None
        } else {
            Some((rng.gen_range(0..self.connections.len()), self.pgen))
        }
    }

    /// Borrow the complete relative connection topology for numerical batching.
    /// # Arguments:
    /// - `self`: Persistent uniform connection generator.
    /// # Returns:
    /// - `&[OrthogonalConnection]`: Connection table indexed by compact spawn requests.
    pub(in crate::stochastic) fn connections(&self) -> &[OrthogonalConnection] {
        &self.connections
    }
}

/// Evaluate the shifted off-diagonal coupling
/// `T_{xw}(\Delta\tau) = H_{xw} - E_s(\Delta \tau) S_{xw}.`
/// # Arguments:
/// - `lambda`: Child determinant index `x`.
/// - `gamma`: Source determinant index `w`.
/// - `shift`: Current population-control shift `E_s`.
/// - `data`: Immutable stochastic propagation data.
/// - `scratch`: Scratch space for nonorthogonal Wick quantities.
/// # Returns:
/// - `f64`: Shifted coupling `T_{xw}`.
pub(in crate::stochastic) fn coupling(
    lambda: usize,
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    let lambda_state = data.space.state(crate::noci::NOCIIndex(lambda));
    let gamma_state = data.space.state(crate::noci::NOCIIndex(gamma));
    let (lambda_oa, lambda_ob) = data.space.occupations(crate::noci::NOCIIndex(lambda));
    let (gamma_oa, gamma_ob) = data.space.occupations(crate::noci::NOCIIndex(gamma));

    if lambda_state.parent == gamma_state.parent
        && (lambda_oa ^ gamma_oa).count_ones() + (lambda_ob ^ gamma_ob).count_ones() > 4
    {
        return 0.0;
    }

    let (hxw, sxw) = find_hs(data, lambda, gamma, scratch);

    hxw - shift * sxw
}

/// Construct exact heat-bath excitation-generation data for determinant
/// `w. For every x \neq w, the heat-bath weight is`
/// `w_{xw} = |T_{xw}(\Delta \tau)|. The total weight is`
/// `W_w = \sum_{x \neq w}w_{xw}.`
/// # Arguments:
/// - `gamma`: Parent determinant index w.
/// - `shift`: `Current population-control shift E_s(\Delta \tau).`
/// - `data`: Immutable stochastic propagation data.
/// - `scratch`: Scratch space for nonorthogonal Wick quantities.
/// # Returns:
/// - `HeatBath`: Couplings and cumulative weights for sampling children.
pub(in crate::stochastic) fn init_heat_bath(
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> HeatBath {
    let ndets = data.space.len();
    // Total weight W_w = \sum_{x != w} |T_{x w}(\Delta \tau)|.
    let mut sumxw = 0.0_f64;
    // Cumulative weights A_n = \sum_{i = 1}^n |T_{i w}(\Delta \tau)|.
    let mut cumulatives: Vec<f64> = Vec::new();
    // Corresponding child indices to the cumulatives.
    let mut lambdas: Vec<usize> = Vec::new();
    // Signed shifted couplings T_{x w}(\Delta \tau).
    let mut ks: Vec<f64> = Vec::new();

    cumulatives.reserve(ndets - 1);
    lambdas.reserve(ndets - 1);
    ks.reserve(ndets - 1);

    for lambda in 0..ndets {
        if lambda == gamma {
            continue;
        }
        let k = coupling(lambda, gamma, shift, data, scratch);

        sumxw += k.abs();
        cumulatives.push(sumxw);
        lambdas.push(lambda);
        ks.push(k);
    }
    HeatBath {
        sumxw,
        cumulatives,
        lambdas,
        ks,
    }
}

/// Sample an off-diagonal child determinant from the exact heat-bath
/// `distribution. For nonzero total weight, P_{\mathrm{gen}}(xw)`
/// `= |T_{xw}(\Delta \tau)| / W_w, if W_w = 0,`
/// the function falls back to uniform sampling.
/// # Arguments:
/// - `gamma`: Parent determinant index `w`.
/// - `shift`: Current population-control shift `E_s`, used by the
///   uniform fallback.
/// - `data`: Immutable stochastic propagation data.
/// - `rng`: Random-number generator.
/// - `hb`: Exact heat-bath data constructed for the same determinant
///   and shift.
/// - `scratch`: Scratch space used for nonorthogonal Wick's quantities.
/// # Returns:
/// - `(f64, f64, usize)`: `Generation probability P_{\mathrm{Gen}}(x|w),`
///   `shifted coupling T_{xw}(\Delta \tau), and sampled child index x.`
pub(in crate::stochastic) fn pgen_heat_bath(
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    rng: &mut QmcRng,
    hb: &HeatBath,
    scratch: &mut WickScratchSpin<f64>,
) -> (f64, f64, usize) {
    let ndets = data.space.len();
    // If \Sum_{x \neq w} |H_{xw} - E_s^S(\tau)S_{xw} (sumxw)
    // is zero (unsure how likely this is) then fallback to uniform distribution.
    if hb.sumxw == 0.0 {
        let mut lambda = rng.gen_range(0..(ndets - 1));
        if lambda >= gamma {
            lambda += 1;
        }
        let k = coupling(lambda, gamma, shift, data, scratch);
        let pgen = 1.0 / ((ndets - 1) as f64);
        return (pgen, k, lambda);
    }

    // We want P_{\text{gen}} = |H_{xw} - E_s^S(\tau)S_{xw}| /
    // \Sum_{x \neq w} |H_{xw} - E_s^S(\tau)S_{xw}|. We
    // choose a number (target) uniformly in \Sum_{x \neq w} |H_{xw} - E_s^S(\tau)S_{xw}
    // (sumxw) and define the sequence of cumulative sums:
    //      A_1 = |H_{1w} - E_s^S(\tau)S_{1w}|
    //      ..
    //      A_n = \sum_{i=1}^n |H_{iw} - E_s^S(\tau)S_{iw}|.
    // The probability that target is between A_{j-1} and A_{j} is:
    //     |H_{xw} - E_s^S(\tau)S_{xw}| /
    //     \Sum_{x \neq w} |H_{xw} - E_s^S(\tau)S_{xw}|,
    // which is exactly the distribution we want to sample. We can therefore add the A's
    // until we pass the target at which point the last tested x is the one chosen
    // with the correct probability, and we can compute k and pgen accordingly.
    let target = rng.gen_range(0.0..hb.sumxw);
    // Find first index where the cumulative sum is more than the target. Binary search returns
    // Result <usize, usize> where Ok(i) is element exactly equal to target and Err(i) is insertion
    // index where target would be inserted to keep array sorted. In both cases this is what we
    // want.
    let i = match hb
        .cumulatives
        .binary_search_by(|x| x.partial_cmp(&target).unwrap())
    {
        Ok(i) => i,
        Err(i) => i,
    };

    // Return P_{\text{gen}}, H_{xw} - E_s^S(\tau)S_{xw} and sampled child index.
    let lambda = hb.lambdas[i];
    let k = hb.ks[i];
    let pgen = k.abs() / hb.sumxw;
    (pgen, k, lambda)
}

/// Apply one report-level stochastic Newton update to the overlap mixture probability.
/// The accumulated derivatives estimate `M_2'(p)` and `M_2''(p)` for
/// `M_2(p) = \sum_w A_{wx}^2/q_p(w|x)`, so the Newton update is
/// `p_{\text{next}} = p - M_2'(p)/M_2''(p)`.
/// # Arguments:
/// - `state`: Current propagation state containing the mutable overlap mixture probability.
/// - `derivatives`: Report-local derivative sums to reduce and then clear.
/// - `input`: User input options.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `()`: Updates `state.overlap_weight` for the next report when the Newton step is finite.
pub(in crate::stochastic) fn update_overlap_weight(
    state: &mut PropagationState,
    derivatives: &mut OverlapDerivativeSums,
    input: &Input,
    run: &QMCRunInfo,
    world: &impl Communicator,
) {
    let qmc = input.qmc.as_ref().unwrap();
    if !qmc.optimise_overlap_weight || qmc.excitation_gen != ExcitationGen::OverlapWeighted {
        *derivatives = OverlapDerivativeSums::default();
        return;
    }

    let local = [derivatives.gradient, derivatives.hessian];
    let mut global = [0.0; 2];
    if run.nranks == 1 {
        global = local;
    } else {
        world.all_reduce_into(&local, &mut global, SystemOperation::sum());
    }

    let gradient = global[0];
    let hessian = global[1];
    if gradient.is_finite() && hessian.is_finite() && hessian > 0.0 {
        // Convex stochastic Newton step:
        // \Delta p = -G/H, with G \approx M_2'(p) and H \approx M_2''(p).
        let delta = -gradient / hessian;
        if delta.is_finite() {
            let candidate = state.overlap_weight + delta;
            state.overlap_weight = if candidate < 0.0 {
                0.0
            } else if candidate >= 1.0 {
                1.0 - f64::EPSILON
            } else {
                candidate
            };
        }
    }

    *derivatives = OverlapDerivativeSums::default();
}
