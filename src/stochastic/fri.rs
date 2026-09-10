// stochastic/fri.rs

// External crate imports.
use rand::Rng;

// Parent/sibling imports.
use super::state::{PopulationUpdate, QmcRng};

/// Apply unbiased FRI stochastic rounding to one signed value.
/// For cutoff `c > 0`, `\Phi_c(x) = x` when `|x| >= c`, while for
/// `0 < |x| < c`, `\Phi_c(x) = sign(x)c` with probability `|x|/c` and zero otherwise.
/// Hence `E[\Phi_c(x) | x] = x`.
/// # Arguments:
/// - `value`: Signed value `x`.
/// - `cutoff`: FRI amplitude cutoff `c`; nonpositive values disable compression.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `f64`: Unbiased stochastically rounded value.
pub(in crate::stochastic) fn round(
    value: f64,
    cutoff: f64,
    rng: &mut QmcRng,
) -> f64 {
    // Values outside the stochastic interval are retained exactly,
    // `\Phi_c(0) = 0`,
    // `\Phi_c(x) = x` for `|x| >= c`.
    if value == 0.0 || cutoff <= 0.0 || value.abs() >= cutoff {
        return value;
    }

    // For `0 < |x| < c`, retain `sign(x)c` with probability `p = |x|/c`, giving
    // `E[\Phi_c(x)] = p sign(x)c = x`.
    if rng.r#gen::<f64>() < value.abs() / cutoff {
        cutoff.copysign(value)
    } else {
        0.0
    }
}

/// Apply FRI compression to sparse population updates in place.
/// Each stored amplitude obeys `E[\Phi_c(x) | x] = x`.
/// # Arguments:
/// - `updates`: Sparse population-update vector.
/// - `cutoff`: FRI amplitude cutoff.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `()`: Replaces `updates` with the retained FRI sample.
pub(in crate::stochastic) fn compress_sparse(
    updates: &mut Vec<PopulationUpdate>,
    cutoff: f64,
    rng: &mut QmcRng,
) {
    // Compact in place while applying `x_i -> \Phi_c(x_i)` so no second sparse allocation is
    // required.
    let mut out = 0usize;
    for i in 0..updates.len() {
        let mut update = updates[i];
        update.dn = round(update.dn, cutoff, rng);
        if update.dn != 0.0 {
            updates[out] = update;
            out += 1;
        }
    }
    updates.truncate(out);
}

/// Select the FRI cutoff giving a requested expected retained NNZ.
/// For amplitudes `x_i`, `M(c) = \sum_i min(1, |x_i|/c)`. The active deterministic set
/// `D(c) = {i : |x_i| >= c}` gives
/// `c = \sum_{i \notin D}|x_i|/(M_target - |D|)` at a self-consistent solution.
/// # Arguments:
/// - `values`: Dense or sparse source representation.
/// - `target_nnz`: Requested expected retained nonzero count.
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
    let tolerance = (0.005 * target).max(1.0);
    let mut low = minimum;
    let mut high = safe_cutoff;
    for _ in 0..16 {
        let midpoint = (0.5 * (low.ln() + high.ln())).exp();
        let expected = expected_nnz(midpoint);
        if (expected - target).abs() <= tolerance {
            return midpoint;
        }
        if expected > target {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    (0.5 * (low.ln() + high.ln())).exp()
}

/// Compress a dense real vector directly into sparse population updates.
/// The dense input is cleared while scanning, avoiding a temporary uncompressed sparse copy.
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
    updates: &mut Vec<PopulationUpdate>,
) {
    updates.clear();
    // B is nearly dense before report FRI, so combine stochastic rounding, dense-buffer clearing and
    // sparse materialisation in one O(N_det) pass.
    for (det, value) in values.iter_mut().enumerate() {
        let dn = round(*value, cutoff, rng);
        *value = 0.0;
        if dn != 0.0 {
            updates.push(PopulationUpdate {
                det: det as u64,
                dn,
            });
        }
    }
}
