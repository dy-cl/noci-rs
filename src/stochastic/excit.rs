// stochastic/excit.rs
// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::Rng;

// Crate-root imports.
use crate::input::{ExcitationGen, Input};
use crate::noci::{MOCache, NOCIData};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::{DetState, Excitation, ExcitationSpin, ReducedTwoSpinState};

// Parent/sibling imports.
use super::common::find_hs;
use super::state::{HeatBath, OverlapDerivativeSums, PropagationState, QMCRunInfo, QmcRng};

/// Return the orbital index of the `rank`th set bit.
/// Callers guarantee `rank < bits.count_ones()`.
/// # Arguments:
/// - `bits`: Orbital bit mask.
/// - `rank`: Zero-based rank among set bits.
/// # Returns
/// - `usize`: Selected orbital index.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
fn select_set_bit(
    bits: u128,
    rank: usize,
) -> usize {
    use std::arch::x86_64::_pdep_u64;

    let low = bits as u64;
    let nlow = low.count_ones() as usize;
    if rank < nlow {
        let selected = unsafe { _pdep_u64(1u64 << rank, low) };
        selected.trailing_zeros() as usize
    } else {
        let high = (bits >> 64) as u64;
        let selected = unsafe { _pdep_u64(1u64 << (rank - nlow), high) };
        64 + selected.trailing_zeros() as usize
    }
}

/// Return the orbital index of the `rank`th set bit using the portable clear-lowest-bit path.
/// Callers guarantee `rank < bits.count_ones()`.
/// # Arguments:
/// - `bits`: Orbital bit mask.
/// - `rank`: Zero-based rank among set bits.
/// # Returns
/// - `usize`: Selected orbital index.
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline(always)]
fn select_set_bit(
    mut bits: u128,
    mut rank: usize,
) -> usize {
    // Remove the `rank` lowest set bits. The next trailing set bit is the selected orbital.
    while rank != 0 {
        bits &= bits - 1;
        rank -= 1;
    }

    bits.trailing_zeros() as usize
}

/// Relative occupied/virtual-rank topology of one orthogonal Hamiltonian connection.
#[derive(Clone, Copy, Debug)]
pub(in crate::stochastic) enum OrthogonalConnection {
    /// Replace one alpha electron.
    AlphaSingle {
        /// Rank of the removed orbital among occupied alpha orbitals.
        occupied: u8,
        /// Rank of the inserted orbital among virtual alpha orbitals.
        virtual_: u8,
    },
    /// Replace one beta electron.
    BetaSingle {
        /// Rank of the removed orbital among occupied beta orbitals.
        occupied: u8,
        /// Rank of the inserted orbital among virtual beta orbitals.
        virtual_: u8,
    },
    /// Replace two alpha electrons.
    AlphaDouble {
        /// Ranks of the removed orbitals among occupied alpha orbitals.
        occupied: [u8; 2],
        /// Ranks of the inserted orbitals among virtual alpha orbitals.
        virtual_: [u8; 2],
    },
    /// Replace two beta electrons.
    BetaDouble {
        /// Ranks of the removed orbitals among occupied beta orbitals.
        occupied: [u8; 2],
        /// Ranks of the inserted orbitals among virtual beta orbitals.
        virtual_: [u8; 2],
    },
    /// Replace one alpha and one beta electron.
    AlphaBetaDouble {
        /// Rank of the removed orbital among occupied alpha orbitals.
        occupied_a: u8,
        /// Rank of the inserted orbital among virtual alpha orbitals.
        virtual_a: u8,
        /// Rank of the removed orbital among occupied beta orbitals.
        occupied_b: u8,
        /// Rank of the inserted orbital among virtual beta orbitals.
        virtual_b: u8,
    },
}

/// Persistent uniform proposal topology for parent-orthogonal Hamiltonian connections.
pub(in crate::stochastic) struct OrthogonalUniformGenerator {
    /// Valid alpha-orbital mask shared by every parent basis.
    alpha_mask: u128,
    /// Valid beta-orbital mask shared by every parent basis.
    beta_mask: u128,
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
    /// - `source`: Representative determinant defining fixed electron counts.
    /// - `cache`: Representative MO cache defining common alpha and beta orbital dimensions.
    /// # Returns
    /// - `Self`: System-wide flat uniform connection table and valid-orbital masks.
    pub(in crate::stochastic) fn new(
        source: &DetState<f64>,
        cache: &MOCache<f64>,
    ) -> Self {
        let noa = source.oa.count_ones() as usize;
        let nob = source.ob.count_ones() as usize;
        let nva = cache.ha.nrows() - noa;
        let nvb = cache.hb.nrows() - nob;
        let nas = noa * nva;
        let nbs = nob * nvb;
        let naa = (noa * noa.saturating_sub(1) / 2) * (nva * nva.saturating_sub(1) / 2);
        let nbb = (nob * nob.saturating_sub(1) / 2) * (nvb * nvb.saturating_sub(1) / 2);
        let nab = nas * nbs;
        let mut connections = Vec::with_capacity(nas + nbs + naa + nbb + nab);

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

        let alpha_mask = if cache.ha.nrows() == 128 {
            u128::MAX
        } else {
            (1u128 << cache.ha.nrows()) - 1
        };
        let beta_mask = if cache.hb.nrows() == 128 {
            u128::MAX
        } else {
            (1u128 << cache.hb.nrows()) - 1
        };
        let pgen = if connections.is_empty() {
            0.0
        } else {
            1.0 / connections.len() as f64
        };

        Self {
            alpha_mask,
            beta_mask,
            pgen,
            connections,
        }
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
}

/// Resolve one relative connection into `E_{Dx}` and its reduced numerical state.
/// Occupied and virtual ranks are mapped into source-specific labels, after which the existing
/// excitation masks define `O'_\sigma=(O_\sigma\setminus I_\sigma)\cup A_\sigma` downstream.
/// # Arguments:
/// - `generator`: Persistent valid-orbital masks and connection table.
/// - `connection`: Sampled connection-table index.
/// - `source`: Source determinant `|\Phi_x^P\rangle`.
/// # Returns
/// - `(Excitation, ReducedTwoSpinState)`: Existing excitation masks and prepared numerical state.
#[inline(always)]
pub(in crate::stochastic) fn resolve_orthogonal_connection(
    generator: &OrthogonalUniformGenerator,
    connection: usize,
    source: &DetState<f64>,
) -> (Excitation, ReducedTwoSpinState) {
    let vira = generator.alpha_mask & !source.oa;
    let virb = generator.beta_mask & !source.ob;
    let mut excitation = Excitation::empty();

    match generator.connections[connection] {
        OrthogonalConnection::AlphaSingle { occupied, virtual_ } => {
            let i = select_set_bit(source.oa, occupied as usize);
            let a = select_set_bit(vira, virtual_ as usize);
            excitation.alpha = ExcitationSpin {
                holes: 1u128 << i,
                parts: 1u128 << a,
            };
        }
        OrthogonalConnection::BetaSingle { occupied, virtual_ } => {
            let i = select_set_bit(source.ob, occupied as usize);
            let a = select_set_bit(virb, virtual_ as usize);
            excitation.beta = ExcitationSpin {
                holes: 1u128 << i,
                parts: 1u128 << a,
            };
        }
        OrthogonalConnection::AlphaDouble { occupied, virtual_ } => {
            let i = select_set_bit(source.oa, occupied[0] as usize);
            let j = select_set_bit(source.oa, occupied[1] as usize);
            let a = select_set_bit(vira, virtual_[0] as usize);
            let b = select_set_bit(vira, virtual_[1] as usize);
            excitation.alpha = ExcitationSpin {
                holes: (1u128 << i) | (1u128 << j),
                parts: (1u128 << a) | (1u128 << b),
            };
        }
        OrthogonalConnection::BetaDouble { occupied, virtual_ } => {
            let i = select_set_bit(source.ob, occupied[0] as usize);
            let j = select_set_bit(source.ob, occupied[1] as usize);
            let a = select_set_bit(virb, virtual_[0] as usize);
            let b = select_set_bit(virb, virtual_[1] as usize);
            excitation.beta = ExcitationSpin {
                holes: (1u128 << i) | (1u128 << j),
                parts: (1u128 << a) | (1u128 << b),
            };
        }
        OrthogonalConnection::AlphaBetaDouble {
            occupied_a,
            virtual_a,
            occupied_b,
            virtual_b,
        } => {
            let i = select_set_bit(source.oa, occupied_a as usize);
            let a = select_set_bit(vira, virtual_a as usize);
            let j = select_set_bit(source.ob, occupied_b as usize);
            let b = select_set_bit(virb, virtual_b as usize);
            excitation.alpha = ExcitationSpin {
                holes: 1u128 << i,
                parts: 1u128 << a,
            };
            excitation.beta = ExcitationSpin {
                holes: 1u128 << j,
                parts: 1u128 << b,
            };
        }
    }

    let reduced = ReducedTwoSpinState::from_excitation((source.oa, source.ob), &excitation);
    (excitation, reduced)
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
    let lambda_det = &data.basis[lambda];
    let gamma_det = &data.basis[gamma];

    if lambda_det.parent == gamma_det.parent
        && (lambda_det.oa ^ gamma_det.oa).count_ones() + (lambda_det.ob ^ gamma_det.ob).count_ones()
            > 4
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
    let ndets = data.basis.len();
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
    let ndets = data.basis.len();
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
