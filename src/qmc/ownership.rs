// qmc/ownership.rs

/// Map one integer key to a deterministic MPI owner.
/// # Arguments
/// - `key`: Integer key, such as a determinant index.
/// - `nranks`: Number of MPI ranks.
/// # Returns
/// - `usize`: MPI rank that owns the key.
#[inline(always)]
pub(crate) fn owner(
    key: usize,
    nranks: usize,
) -> usize {
    (mix(key as u64) as usize) % nranks
}

/// Map a key of several words, such as occupation words and a reassignment salt, to a
/// deterministic MPI owner by folding the words through the avalanche mix.
/// # Arguments
/// - `words`: Key words.
/// - `nranks`: Number of MPI ranks.
/// # Returns
/// - `usize`: MPI rank that owns the key.
#[inline(always)]
pub(crate) fn owner_words(
    words: &[u64],
    nranks: usize,
) -> usize {
    (words.iter().fold(0u64, |h, &w| mix(h ^ w)) as usize) % nranks
}

/// Apply the deterministic avalanche mix shared by stochastic ownership maps.
/// # Arguments:
/// - `x`: Unsigned key word.
/// # Returns:
/// - `u64`: Mixed key word.
#[inline(always)]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}
