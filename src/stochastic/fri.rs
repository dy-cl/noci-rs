// stochastic/fri.rs
//! Fast randomized iteration compression for stochastic propagation.
//! Fixed-cutoff compression uses prescribed marginals `p_i = \min(1, |x_i|/c)` and therefore has
//! nonintegral total inclusion mass in general. With these marginals and selected values `x_i/p_i`,
//! pivotal and independent Bernoulli sampling have the same coordinatewise expected `L^2` error;
//! pivotal sampling changes selection covariances and can reduce retained-count fluctuations.
//! Target-NNZ compression solves `M(c) = \sum_i \min(1, |x_i|/c) = m`. In exact arithmetic, the
//! deterministic-large-component split and remaining probabilities proportional to magnitudes give
//! `E[\Phi(x)] = x`, `||\Phi(x)||_0 <= m`, and minimum expected `L^2` compression error among
//! unbiased `m`-sparse approximations. Floating cutoff solves can differ from `m` by roundoff, and
//! the generic cutoff compressor does not independently enforce a machine-arithmetic hard cap.
//! The FRI framework follows L.-H. Lim and J. Weare, "Fast Randomized Iteration: Diffusion Monte
//! Carlo through the Lens of Numerical Linear Algebra", SIAM Rev. 59, 547-587 (2017), DOI:
//! 10.1137/15M1040827. Its FCI-FRI application, correlated/systematic sampling predecessor, and
//! empirical variance reductions are described by S. M. Greene, R. J. Webber, J. Weare, and
//! T. C. Berkelbach, "Beyond Walkers in Stochastic Quantum Chemistry: Reducing Error Using Fast
//! Randomized Iteration", J. Chem. Theory Comput. 15, 4834-4850 (2019), DOI:
//! 10.1021/acs.jctc.9b00422; those empirical gains are not a general variance theorem. Pivotal
//! sampling follows Section 5 and Algorithms 5.1-5.2, with fixed-`m` optimality from Proposition
//! 5.2, of
//! J. Weare and R. J. Webber, "Randomly sparsified Richardson iteration: A dimension-independent
//! sparse linear solver", Commun. Pure Appl. Math. 79, 89-122 (2026), DOI: 10.1002/cpa.70012.

// External crate imports.
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

// Crate-root imports.
use crate::time_call;

// Parent/sibling imports.
use super::state::{NOCIPopulationUpdate, QMCRunInfo, QmcRng, SparsePopulations};

/// One unresolved pivotal candidate with fixed selected amplitude `x_i/p_i`.
/// The current probability changes under pivotal pairing, while `selected_amplitude` remains based
/// on the original marginal inclusion probability `p_i`.
#[derive(Clone, Copy)]
struct PivotalCandidate<T: Copy> {
    /// Key or sparse payload retained when this candidate is selected.
    payload: T,
    /// Original selected amplitude `x_i/p_i`.
    selected_amplitude: f64,
    /// Current unresolved pivotal inclusion probability.
    probability: f64,
}

/// Combine one candidate with the current unresolved pivotal state.
/// For probabilities `a,b in (0,1)`, the pairwise update fixes at least one decision while
/// preserving both marginals. If `a+b<1`, one survivor receives probability `a+b`; otherwise one
/// candidate is selected and the other receives probability `a+b-1`. This is Algorithms 5.1-5.2
/// of J. Weare and R. J. Webber, Commun. Pure Appl. Math. 79, 89-122 (2026),
/// DOI: 10.1002/cpa.70012.
/// # Arguments:
/// - `pending`: At most one unresolved candidate from preceding pivotal updates.
/// - `candidate`: Next fractional candidate in the chosen deterministic ordering.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `Option<(T, f64)>`: Newly selected payload and its original `x_i/p_i`, if one was fixed.
fn pivotal_push<T: Copy>(
    pending: &mut Option<PivotalCandidate<T>>,
    candidate: PivotalCandidate<T>,
    rng: &mut QmcRng,
) -> Option<(T, f64)> {
    let Some(first) = pending.take() else {
        *pending = Some(candidate);
        return None;
    };

    let a = first.probability;
    let b = candidate.probability;
    let sum = a + b;

    // For `a+b<1`, carry one unresolved candidate with mass `a+b`; choosing
    // the first with probability `a/(a+b)` preserves both original marginals.
    if sum < 1.0 {
        if rng.r#gen::<f64>() < a / sum {
            *pending = Some(PivotalCandidate {
                probability: sum,
                ..first
            });
        } else {
            *pending = Some(PivotalCandidate {
                probability: sum,
                ..candidate
            });
        }
        return None;
    }

    // For `a+b>=1`, resolve one inclusion now and carry excess mass
    // `a+b-1`; probability `(1-b)/(2-a-b)` preserves the first marginal.
    let select_first = rng.r#gen::<f64>() < (1.0 - b) / (2.0 - sum);
    let remaining_probability = sum - 1.0;
    if select_first {
        if remaining_probability > 0.0 {
            *pending = Some(PivotalCandidate {
                probability: remaining_probability,
                ..candidate
            });
        }
        Some((first.payload, first.selected_amplitude))
    } else {
        if remaining_probability > 0.0 {
            *pending = Some(PivotalCandidate {
                probability: remaining_probability,
                ..first
            });
        }
        Some((candidate.payload, candidate.selected_amplitude))
    }
}

/// Resolve the sole candidate that may remain after pivotal reduction.
/// A Bernoulli draw with its current probability completes the reduction without changing its
/// original selected amplitude, preserving `E[\Phi(x)] = x` for nonintegral inclusion mass.
/// # Arguments:
/// - `pending`: Final unresolved candidate, if any.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `Option<(T, f64)>`: Selected payload and original `x_i/p_i`, if retained.
fn resolve_pivotal<T: Copy>(
    pending: Option<PivotalCandidate<T>>,
    rng: &mut QmcRng,
) -> Option<(T, f64)> {
    if let Some(candidate) = pending
        && rng.r#gen::<f64>() < candidate.probability
    {
        Some((candidate.payload, candidate.selected_amplitude))
    } else {
        None
    }
}

/// Construct a fractional fixed-cutoff candidate for `0 < |x_i| < c`.
/// Its original marginal is `p_i=|x_i|/c`, hence its selected amplitude is exactly
/// `x_i/p_i=sign(x_i)c`.
/// # Arguments:
/// - `payload`: Key or sparse value associated with `x_i`.
/// - `amplitude`: Signed amplitude `x_i`.
/// - `cutoff`: Positive finite cutoff `c`.
/// # Returns:
/// - `PivotalCandidate<T>`: Fractional candidate ready for pivotal reduction.
fn cutoff_candidate<T: Copy>(
    payload: T,
    amplitude: f64,
    cutoff: f64,
) -> PivotalCandidate<T> {
    PivotalCandidate {
        payload,
        selected_amplitude: cutoff.copysign(amplitude),
        probability: amplitude.abs() / cutoff,
    }
}

/// Sparse amplitude interface shared by keyed FRI vectors and spawn-event batches.
pub(in crate::stochastic) trait FriAmplitude: Copy {
    /// Return the signed amplitude to compress.
    /// # Arguments:
    /// - `self`: Sparse update value.
    /// # Returns:
    /// - `f64`: Signed update amplitude.
    fn amplitude(&self) -> f64;

    /// Replace the signed amplitude without changing its sparse key.
    /// # Arguments:
    /// - `self`: Sparse update value.
    /// - `amplitude`: Replacement signed amplitude.
    /// # Returns:
    /// - `()`: Updates the amplitude in place.
    fn set_amplitude(
        &mut self,
        amplitude: f64,
    );
}

impl FriAmplitude for NOCIPopulationUpdate {
    /// Return one NOCI-coordinate population update amplitude.
    /// # Arguments:
    /// - `self`: Sparse population update.
    /// # Returns:
    /// - `f64`: Signed population change.
    fn amplitude(&self) -> f64 {
        self.dn
    }

    /// Replace one NOCI-coordinate population update amplitude.
    /// # Arguments:
    /// - `self`: Sparse population update.
    /// - `amplitude`: Replacement population change.
    /// # Returns:
    /// - `()`: Updates `dn` in place.
    fn set_amplitude(
        &mut self,
        amplitude: f64,
    ) {
        self.dn = amplitude;
    }
}

/// Construct a pivotal FRI sparse sample of persistent populations.
/// Fixed `population_cutoff=c` gives marginals `p_i=\min(1,|N_i|/c)` and selected fractional
/// amplitudes `N_i/p_i=sign(N_i)c`, so `E[\tilde N_i | N]=N_i`. For these fixed marginals,
/// `E[||\tilde N-N||_2^2]=\sum_i N_i^2(1/p_i-1)`, equal to independent Bernoulli sampling.
/// Pivotal dependence instead changes covariances and can reduce retained-count fluctuations; it
/// does not guarantee lower variance for every downstream estimator or reinterpret `c` as a target
/// NNZ.
/// Large vectors reduce independently within Rayon chunks, then merge at most one unresolved
/// candidate per chunk, retaining parallel `O(N/nthreads)` scanning and `O(nthreads)` merge state.
/// # Arguments:
/// - `populations`: Persistent rank-local population vector `N`.
/// - `sampled`: Temporary sparse sampled vector `\tilde N`.
/// - `cutoff`: Stochastic sampling cutoff `c`.
/// - `run`: Rank-local determinant ownership information.
/// - `rng`: Random-number generator.
/// - `chunks`: Reusable parallel sampling buffers.
/// # Returns:
/// - `()`: Replaces `sampled` with sparse unbiased sample of `populations`.
pub(in crate::stochastic) fn sample_populations(
    populations: &[f64],
    sampled: &mut SparsePopulations,
    cutoff: f64,
    run: &QMCRunInfo,
    rng: &mut QmcRng,
    chunks: &mut Vec<Vec<(usize, f64)>>,
) {
    time_call!(crate::timers::stochastic::add_sample_populations, {
        if cutoff <= 0.0 {
            sampled.clear();

            // With compression disabled, `\tilde N = N`, so retain every exactly nonzero local
            // population.
            for (k, &population) in populations.iter().enumerate() {
                if population != 0.0 {
                    sampled.insert_nonzero(run.owned[k], population);
                }
            }

            return;
        }

        if cutoff == f64::INFINITY {
            sampled.clear();
            return;
        }

        if populations.len() < 8192 {
            sampled.clear();
            let mut pending = None;

            for (&det, &population) in run.owned.iter().zip(populations.iter()) {
                if population == 0.0 {
                    continue;
                }
                if population.abs() >= cutoff {
                    sampled.insert_nonzero(det, population);
                } else {
                    // Fractional entries use `p_i=|N_i|/c` and selected value `sign(N_i)c`.
                    if let Some((det, population)) =
                        pivotal_push(&mut pending, cutoff_candidate(det, population, cutoff), rng)
                    {
                        sampled.insert_nonzero(det, population);
                    }
                }
            }
            if let Some((det, population)) = resolve_pivotal(pending, rng) {
                sampled.insert_nonzero(det, population);
            }

            return;
        }

        // Chunks pivotal-reduce in parallel and expose at most one fractional candidate each.
        let nthreads = rayon::current_num_threads().max(1);
        let chunk_size = populations.len().div_ceil(nthreads).max(1024);
        let nchunks = populations.len().div_ceil(chunk_size);
        let seed = rng.r#gen::<u64>();
        let mut pending_chunks = vec![None; nchunks];

        chunks.resize_with(nchunks, Vec::new);
        chunks[..nchunks]
            .par_iter_mut()
            .zip(pending_chunks.par_iter_mut())
            .enumerate()
            .for_each(|(chunk, (entries, pending))| {
                let mut chunk_rng =
                    QmcRng::seed_from_u64(seed ^ (chunk as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let start = chunk * chunk_size;
                let end = (start + chunk_size).min(populations.len());
                let populations = &populations[start..end];
                let owned = &run.owned[start..end];

                entries.clear();
                *pending = None;
                for (&det, &population) in owned.iter().zip(populations.iter()) {
                    if population == 0.0 {
                        continue;
                    }
                    if population.abs() >= cutoff {
                        entries.push((det, population));
                    } else if let Some(selected) = pivotal_push(
                        pending,
                        cutoff_candidate(det, population, cutoff),
                        &mut chunk_rng,
                    ) {
                        entries.push(selected);
                    }
                }
            });

        // Merge chunk residuals in deterministic order; hierarchical pairing preserves marginals.
        sampled.clear();
        for entries in chunks.iter().take(nchunks) {
            for &(det, population) in entries {
                sampled.insert_nonzero(det, population);
            }
        }
        let mut pending = None;
        for candidate in pending_chunks.into_iter().flatten() {
            if let Some((det, population)) = pivotal_push(&mut pending, candidate, rng) {
                sampled.insert_nonzero(det, population);
            }
        }
        if let Some((det, population)) = resolve_pivotal(pending, rng) {
            sampled.insert_nonzero(det, population);
        }
    });
}

/// Apply fixed-cutoff pivotal FRI compression to sparse keyed amplitudes in place.
/// Deterministic entries with `|x_i|>=c` remain unchanged. Fractional entries use
/// `p_i=|x_i|/c` and selected amplitude `x_i/p_i=sign(x_i)c`; pairwise pivotal sampling preserves
/// `E[\Phi_c(x_i)|x_i]=x_i`. Nonpositive `c` removes stored zeros only, while `c=+infinity`
/// retains nothing. Selected values compact in place without a second full sparse allocation.
/// # Arguments:
/// - `updates`: Sparse keyed-amplitude vector.
/// - `cutoff`: FRI amplitude cutoff.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `()`: Replaces `updates` with the retained FRI sample.
pub(in crate::stochastic) fn compress_sparse<T: FriAmplitude>(
    updates: &mut Vec<T>,
    cutoff: f64,
    rng: &mut QmcRng,
) {
    if cutoff == f64::INFINITY {
        updates.clear();
        return;
    }

    // Retain amplitudes above the cutoff exactly; send smaller amplitudes
    // through pivotal pairing with inclusion mass `|x_i| / cutoff`.
    let mut out = 0usize;
    let mut pending = None;

    for i in 0..updates.len() {
        let update = updates[i];
        let amplitude = update.amplitude();
        if amplitude == 0.0 {
            continue;
        }
        if cutoff <= 0.0 || amplitude.abs() >= cutoff {
            updates[out] = update;
            out += 1;
        } else if let Some((mut selected, amplitude)) = pivotal_push(
            &mut pending,
            cutoff_candidate(update, amplitude, cutoff),
            rng,
        ) {
            selected.set_amplitude(amplitude);
            updates[out] = selected;
            out += 1;
        }
    }
    // Resolve the final fractional candidate, then keep only compacted
    // selected events in the original sparse buffer.
    if let Some((mut selected, amplitude)) = resolve_pivotal(pending, rng) {
        selected.set_amplitude(amplitude);
        updates[out] = selected;
        out += 1;
    }

    updates.truncate(out);
}

/// Select the pivotal FRI cutoff targeting a requested retained NNZ.
/// For amplitudes `x_i`, `M(c) = \sum_i \min(1, |x_i|/c)`. The active deterministic set
/// `D(c) = {i : |x_i| >= c}` gives
/// `c = \sum_{i \notin D}|x_i|/(m-|D|)` at a self-consistent solution. Feeding these marginals
/// through pivotal sampling gives `E[\Phi(x)]=x` and `||\Phi(x)||_0<=m` in exact arithmetic when
/// `M(c)=m`, matching Algorithm 5.1 and Proposition 5.2 of Weare and Webber. Floating active-set
/// arithmetic can produce `M(c)=m+epsilon`; because the generic compressor has no `m` argument, a
/// final Bernoulli resolution can then retain `m+1` entries with probability `epsilon`.
/// # Arguments:
/// - `values`: Dense or sparse source representation.
/// - `target_nnz`: Requested retained-count target `m`.
/// - `cutoff_hint`: Previous report cutoff, or zero when unavailable.
/// - `magnitude`: Accessor returning `|x_i|` for one stored value.
/// # Returns:
/// - `f64`: Selected cutoff. Zero retains every nonzero entry exactly.
pub(in crate::stochastic) fn target_cutoff<T, F>(
    values: &[T],
    target_nnz: usize,
    cutoff_hint: f64,
    magnitude: F,
) -> f64
where
    F: Fn(&T) -> f64 + Copy,
{
    if target_nnz == 0 {
        return f64::INFINITY;
    }

    // Use the previous report cutoff only as an active-set guess; correctness does not depend on
    // the hint.
    let hint = (cutoff_hint.is_finite() && cutoff_hint > 0.0).then_some(cutoff_hint);
    let mut nonzero = 0usize;
    let mut norm = 0.0;
    let mut minimum = f64::INFINITY;
    let mut deterministic = 0usize;
    let mut stochastic_norm = 0.0;
    let mut largest_stochastic = 0.0f64;
    let mut smallest_deterministic = f64::INFINITY;

    // For a candidate c split the amplitudes into D(c) and its stochastic complement. At a
    // self-consistent active set, `M_target = |D| + \sum_{i \notin D}|x_i|/c`.
    for value in values {
        let amplitude = magnitude(value);
        if amplitude == 0.0 {
            continue;
        }
        nonzero += 1;
        norm += amplitude;
        minimum = minimum.min(amplitude);
        if let Some(cutoff) = hint {
            if amplitude >= cutoff {
                deterministic += 1;
                smallest_deterministic = smallest_deterministic.min(amplitude);
            } else {
                stochastic_norm += amplitude;
                largest_stochastic = largest_stochastic.max(amplitude);
            }
        }
    }

    if nonzero == 0 || target_nnz >= nonzero {
        return 0.0;
    }

    let target = target_nnz as f64;
    let safe_cutoff = norm / target;

    // Test the preceding report's active set first. Self consistency requires
    // `max_{i \notin D}|x_i| <= c <= min_{i \in D}|x_i|`.
    if hint.is_some() && deterministic < target_nnz {
        let cutoff = stochastic_norm / (target - deterministic as f64);
        if cutoff.is_finite()
            && cutoff > 0.0
            && largest_stochastic <= cutoff
            && cutoff <= smallest_deterministic
        {
            return cutoff;
        }
    }

    let mut cutoff = safe_cutoff;

    // Iterate the active-set equation
    // `c_{n+1} = ||x_stochastic||_1/(M_target-|D_n|)`.
    for _ in 0..8 {
        let (deterministic, stochastic_norm, largest_stochastic, smallest_deterministic) =
            values.iter().fold(
                (0usize, 0.0, 0.0f64, f64::INFINITY),
                |(count, norm, largest, smallest), value| {
                    let amplitude = magnitude(value);
                    if amplitude >= cutoff {
                        (count + 1, norm, largest, smallest.min(amplitude))
                    } else {
                        (count, norm + amplitude, largest.max(amplitude), smallest)
                    }
                },
            );
        if deterministic >= target_nnz {
            cutoff = safe_cutoff;
            continue;
        }
        let next = stochastic_norm / (target - deterministic as f64);
        if !next.is_finite() || next <= 0.0 {
            break;
        }
        if largest_stochastic <= next && next <= smallest_deterministic {
            return next;
        }
        cutoff = next;
    }

    // `M(c)` is monotone decreasing, so use bounded log-space bisection only as a pathological
    // fallback when the active-set iteration has not converged.
    let expected_nnz = |cutoff: f64| {
        values
            .iter()
            .map(|value| {
                let amplitude = magnitude(value);
                if amplitude == 0.0 {
                    0.0
                } else {
                    (amplitude / cutoff).min(1.0)
                }
            })
            .sum::<f64>()
    };
    let mut low = minimum;
    let mut high = safe_cutoff;
    for _ in 0..32 {
        let midpoint = (0.5 * (low.ln() + high.ln())).exp();
        let expected = expected_nnz(midpoint);
        if expected > target {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    // Upper bracket keeps the fallback's computed `M(c)` on the non-overshooting side of `m`.
    high
}

/// Compress a dense real vector directly into sparse updates with pivotal FRI.
/// Marginals are `p_i=\min(1,|x_i|/c)`, selected values are `x_i/p_i`, and therefore
/// `E[\Phi_c(x)]=x`. The dense input is cleared during its single `O(N)` scan, avoiding an
/// uncompressed sparse copy and retaining only sparse output plus one unresolved candidate.
/// # Arguments:
/// - `values`: Dense vector to compress and clear.
/// - `cutoff`: FRI amplitude cutoff.
/// - `rng`: Random-number generator.
/// - `updates`: Sparse output buffer.
/// # Returns:
/// - `()`: Clears `values` and replaces `updates` with `\Phi_c(values)`.
pub(in crate::stochastic) fn compress_dense_to_sparse(
    values: &mut [f64],
    cutoff: f64,
    rng: &mut QmcRng,
    updates: &mut Vec<NOCIPopulationUpdate>,
) {
    updates.clear();
    if cutoff == f64::INFINITY {
        values.fill(0.0);
        return;
    }

    let mut pending = None;

    // Combine pivotal reduction, dense-buffer clearing, and sparse materialisation in one scan.
    for (det, value) in values.iter_mut().enumerate() {
        let amplitude = *value;
        *value = 0.0;
        if amplitude == 0.0 {
            continue;
        }
        if cutoff <= 0.0 || amplitude.abs() >= cutoff {
            updates.push(NOCIPopulationUpdate {
                det: det as u64,
                dn: amplitude,
            });
        } else if let Some((det, dn)) = pivotal_push(
            &mut pending,
            cutoff_candidate(det as u64, amplitude, cutoff),
            rng,
        ) {
            updates.push(NOCIPopulationUpdate { det, dn });
        }
    }
    if let Some((det, dn)) = resolve_pivotal(pending, rng) {
        updates.push(NOCIPopulationUpdate { det, dn });
    }
}
