// qmc/rng.rs

// External crate imports.
use rand::SeedableRng;

/// Golden-ratio multiplier decorrelating consecutive stream indices.
const GOLDEN: u64 = 0x9E3779B97F4A7C15;

/// Thirty-two-bit golden-ratio stride separating the seeds of consecutive MPI ranks.
const RANKSTRIDE: u64 = 0x9E3779B9;

/// Stable RNG used by seeded QMC streams.
pub(crate) type QmcRng = rand_xoshiro::Xoshiro256PlusPlus;

/// Derive the seed of one MPI rank from the base seed of the run, `s_r = s_0 + r \phi_{32}`.
/// # Arguments:
/// - `base`: Base seed of the full run.
/// - `irank`: MPI rank.
/// # Returns:
/// - `u64`: Rank-specific seed.
pub(crate) fn rank_seed(
    base: u64,
    irank: usize,
) -> u64 {
    base.wrapping_add((irank as u64).wrapping_mul(RANKSTRIDE))
}

/// Seed one independent stream `s \oplus k \oplus i\phi` from a seed `s`, a salt `k` naming the
/// stream family and an index `i`, such as an iteration or a thread.
/// # Arguments:
/// - `seed`: Rank or run seed.
/// - `salt`: Constant distinguishing independent stream families.
/// - `index`: Stream index within the family.
/// # Returns:
/// - `QmcRng`: Seeded random-number generator.
pub(crate) fn stream(
    seed: u64,
    salt: u64,
    index: u64,
) -> QmcRng {
    QmcRng::seed_from_u64(seed ^ salt ^ index.wrapping_mul(GOLDEN))
}
