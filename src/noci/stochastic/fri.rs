// noci/stochastic/fri.rs
//! Fast Randomised Iteration sampling of the dense NOCIQMC population vectors. The generic
//! pivotal compression and its references live in `crate::qmc`.

// External crate imports.
use rand::Rng;
use rayon::prelude::*;

// Crate-root imports.
use crate::qmc::{FriAmplitude, QmcRng, cutoff_candidate, pivotal_push, resolve_pivotal, stream};
use crate::time_call;

// Parent/sibling imports.
use super::state::{NOCIPopulationUpdate, QMCRunInfo, SparsePopulations};

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
pub(in crate::noci::stochastic) fn sample_populations(
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
                let mut chunk_rng = stream(seed, 0, chunk as u64);
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
pub(in crate::noci::stochastic) fn compress_dense_to_sparse(
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
