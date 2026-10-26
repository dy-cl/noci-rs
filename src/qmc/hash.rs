// qmc/hash.rs

// Standard library imports.
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Multiplier of the word-at-a-time hash.
const SEED: u64 = 0x517cc1b727220a95;

/// Fast deterministic hasher for integer keys: every word `w` updates the state as
/// `h \leftarrow (\operatorname{rotl}_5(h) \oplus w)\,c`. It carries no per-process random key,
/// so every MPI rank iterates equal maps in the same order.
/// # References
/// - The Rust compiler's `FxHasher`, after the Firefox hash.
#[derive(Clone, Copy, Default)]
pub(crate) struct FastHasher(u64);

impl FastHasher {
    /// Mix one word into the state.
    /// # Arguments:
    /// - `self`: Hasher.
    /// - `word`: Word to mix.
    /// # Returns:
    /// - `()`: Updates the state.
    #[inline(always)]
    fn add(
        &mut self,
        word: u64,
    ) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FastHasher {
    /// Return the hash.
    /// # Arguments:
    /// - `self`: Hasher.
    /// # Returns:
    /// - `u64`: Current state.
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    /// Mix arbitrary bytes, eight at a time.
    /// # Arguments:
    /// - `self`: Hasher.
    /// - `bytes`: Bytes to mix.
    /// # Returns:
    /// - `()`: Updates the state.
    fn write(
        &mut self,
        bytes: &[u8],
    ) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    /// Mix one 64-bit word.
    /// # Arguments:
    /// - `self`: Hasher.
    /// - `i`: Word.
    /// # Returns:
    /// - `()`: Updates the state.
    #[inline(always)]
    fn write_u64(
        &mut self,
        i: u64,
    ) {
        self.add(i);
    }

    /// Mix one pointer-sized word.
    /// # Arguments:
    /// - `self`: Hasher.
    /// - `i`: Word.
    /// # Returns:
    /// - `()`: Updates the state.
    #[inline(always)]
    fn write_usize(
        &mut self,
        i: usize,
    ) {
        self.add(i as u64);
    }

    /// Mix one 128-bit word as two 64-bit words.
    /// # Arguments:
    /// - `self`: Hasher.
    /// - `i`: Word.
    /// # Returns:
    /// - `()`: Updates the state.
    #[inline(always)]
    fn write_u128(
        &mut self,
        i: u128,
    ) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }
}

/// Hash map with the fast deterministic hasher.
pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
