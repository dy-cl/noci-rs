// determinant/bits.rs
//! Fixed-width orbital bitstrings. Bit `p` of a [`Bits`] marks spin orbital `p` of one spin, so a
//! bitstring holds an occupation or the holes or particles of an excitation. The default build
//! stores two 64-bit words, 128 orbitals; the `wide` feature stores four, 256 orbitals.
//!
//! Hashing and ordering match those of the `u128` the default width replaces: words are hashed from
//! the lowest orbital up and compared from the highest orbital down.

// Standard library imports.
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

// External crate imports.
use serde::{Deserialize, Serialize};

/// Number of 64-bit words in a bitstring.
#[cfg(not(feature = "wide"))]
const WORDS: usize = 2;
#[cfg(feature = "wide")]
const WORDS: usize = 4;

/// Set of spin orbitals of one spin, with orbital `p` at bit `p % 64` of word `p / 64`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Bits([u64; WORDS]);

/// Orbitals of a bitstring in increasing order.
pub struct Ones(Bits);

impl Bits {
    /// Number of orbitals a bitstring can hold.
    pub const CAPACITY: usize = 64 * WORDS;

    /// Bitstring with no orbitals.
    pub const EMPTY: Bits = Bits([0; WORDS]);

    /// Single orbital `{p}`.
    /// # Arguments:
    /// - `p`: Orbital index, below [`Bits::CAPACITY`], as any unsigned label type.
    /// # Returns
    /// - `Bits`: Bitstring holding only `p`.
    #[inline(always)]
    pub fn bit(p: impl Into<usize>) -> Self {
        let p = p.into();
        let mut words = [0; WORDS];
        words[p >> 6] = 1 << (p & 63);
        Bits(words)
    }

    /// Orbitals `{q : q < p}`.
    /// # Arguments:
    /// - `p`: Exclusive upper bound, at most [`Bits::CAPACITY`].
    /// # Returns
    /// - `Bits`: Bitstring of every orbital below `p`.
    #[inline(always)]
    pub fn below(p: usize) -> Self {
        let mut words = [0; WORDS];
        for (k, word) in words.iter_mut().enumerate() {
            let lo = 64 * k;
            *word = if p >= lo + 64 {
                u64::MAX
            } else if p > lo {
                (1 << (p - lo)) - 1
            } else {
                0
            };
        }
        Bits(words)
    }

    /// Whether orbital `p` is in the set.
    /// # Arguments:
    /// - `p`: Orbital index, below [`Bits::CAPACITY`].
    /// # Returns
    /// - `bool`: Whether bit `p` is set.
    #[inline(always)]
    pub fn contains(
        self,
        p: usize,
    ) -> bool {
        (self.0[p >> 6] >> (p & 63)) & 1 == 1
    }

    /// Number of orbitals in the set.
    /// # Returns
    /// - `u32`: Population count.
    #[inline(always)]
    pub fn count_ones(self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    /// Number of orbitals in the set below `p`, `|\{q \in self : q < p\}|`, whose parity is the
    /// fermionic sign of moving an operator past them.
    /// # Arguments:
    /// - `p`: Exclusive upper bound, below [`Bits::CAPACITY`].
    /// # Returns
    /// - `u32`: Population count below `p`.
    #[inline(always)]
    pub fn count_below(
        self,
        p: usize,
    ) -> u32 {
        let full = p >> 6;
        let mut count = self.0[..full].iter().map(|w| w.count_ones()).sum::<u32>();
        if p & 63 != 0 {
            count += (self.0[full] & ((1 << (p & 63)) - 1)).count_ones();
        }
        count
    }

    /// Whether the set is empty.
    /// # Returns
    /// - `bool`: Whether no bit is set.
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// Lowest orbital of a non-empty set.
    /// # Returns
    /// - `usize`: Index of the lowest set bit.
    /// # Panics
    /// - Panics if the set is empty.
    #[inline(always)]
    pub fn first(self) -> usize {
        let k = self
            .0
            .iter()
            .position(|&w| w != 0)
            .expect("empty bitstring");
        64 * k + self.0[k].trailing_zeros() as usize
    }

    /// Highest orbital of a non-empty set.
    /// # Returns
    /// - `usize`: Index of the highest set bit.
    /// # Panics
    /// - Panics if the set is empty.
    #[inline(always)]
    pub fn last(self) -> usize {
        let k = self
            .0
            .iter()
            .rposition(|&w| w != 0)
            .expect("empty bitstring");
        64 * k + 63 - self.0[k].leading_zeros() as usize
    }

    /// The set without its lowest orbital, `x \& (x - 1)`.
    /// # Returns
    /// - `Bits`: Bitstring with the lowest set bit cleared.
    #[inline(always)]
    pub fn without_first(self) -> Self {
        let mut words = self.0;
        if let Some(word) = words.iter_mut().find(|w| **w != 0) {
            *word &= *word - 1;
        }
        Bits(words)
    }

    /// The 64-bit words from the lowest orbital up, without trailing zero words beyond the first
    /// two, so fingerprints match those of a `u128` and do not depend on the width of the build.
    /// # Returns
    /// - `&[u64]`: Significant words of the bitstring.
    #[inline(always)]
    pub fn words(&self) -> &[u64] {
        let n = self
            .0
            .iter()
            .rposition(|&w| w != 0)
            .map_or(0, |k| k + 1)
            .max(2);
        &self.0[..n]
    }

    /// Orbitals of the set in increasing order.
    /// # Returns
    /// - `Ones`: Iterator over the set bits.
    #[inline(always)]
    pub fn ones(self) -> Ones {
        Ones(self)
    }
}

impl Iterator for Ones {
    type Item = usize;

    /// Next orbital of the set.
    /// # Returns
    /// - `Option<usize>`: Lowest remaining orbital, or `None` once the set is exhausted.
    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.0.is_empty() {
            return None;
        }
        let p = self.0.first();
        self.0 = self.0.without_first();
        Some(p)
    }
}

impl Hash for Bits {
    /// Hash the words from the lowest orbital up, as a `u128` is hashed.
    /// # Arguments:
    /// - `state`: Hasher receiving the words.
    fn hash<H: Hasher>(
        &self,
        state: &mut H,
    ) {
        for &word in &self.0 {
            state.write_u64(word);
        }
    }
}

impl Ord for Bits {
    /// Compare as unsigned integers, from the highest orbital down, as `u128`s compare.
    /// # Arguments:
    /// - `other`: Bitstring to compare with.
    /// # Returns
    /// - `Ordering`: Integer ordering of the two bitstrings.
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}

impl PartialOrd for Bits {
    /// Compare as unsigned integers, from the highest orbital down.
    /// # Arguments:
    /// - `other`: Bitstring to compare with.
    /// # Returns
    /// - `Option<Ordering>`: Integer ordering of the two bitstrings.
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Implement a word-wise binary operator and its assigning form for [`Bits`].
macro_rules! wordwise {
    ($trait:ident, $method:ident, $assign:ident, $assign_method:ident, $op:tt) => {
        impl $trait for Bits {
            type Output = Bits;

            #[inline(always)]
            fn $method(
                mut self,
                rhs: Bits,
            ) -> Bits {
                self.$assign_method(rhs);
                self
            }
        }

        impl $assign for Bits {
            #[inline(always)]
            fn $assign_method(
                &mut self,
                rhs: Bits,
            ) {
                for (a, b) in self.0.iter_mut().zip(rhs.0) {
                    *a $op b;
                }
            }
        }
    };
}

wordwise!(BitAnd, bitand, BitAndAssign, bitand_assign, &=);
wordwise!(BitOr, bitor, BitOrAssign, bitor_assign, |=);
wordwise!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^=);

impl Not for Bits {
    type Output = Bits;

    /// Complement over all [`Bits::CAPACITY`] orbitals.
    /// # Returns
    /// - `Bits`: Bitstring with every bit flipped.
    #[inline(always)]
    fn not(self) -> Bits {
        Bits(self.0.map(|w| !w))
    }
}
