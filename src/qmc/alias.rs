// qmc/alias.rs

// External crate imports.
use rand::Rng;

// Parent/sibling imports.
use super::QmcRng;

/// Alias table drawing index `i` with probability `w_i/\sum_j w_j` in constant time from one
/// uniform number. It is built in `O(n)` from the weights of the occupied entries only, so it is
/// as sparse as the population that defines it.
/// # References
/// - A. J. Walker, *ACM Trans. Math. Softw.* **3**, 253 (1977),
///   [doi:10.1145/355744.355749](https://doi.org/10.1145/355744.355749).
/// - M. D. Vose, *IEEE Trans. Softw. Eng.* **17**, 972 (1991),
///   [doi:10.1109/32.92917](https://doi.org/10.1109/32.92917).
pub(crate) struct AliasTable {
    /// Probability of keeping column `i` rather than its alias.
    keep: Vec<f64>,
    /// Alias of column `i`.
    alias: Vec<usize>,
    /// Sum of the weights.
    total: f64,
}

impl AliasTable {
    /// Build the table of non-negative weights by Vose's method.
    /// # Arguments:
    /// - `weights`: Non-negative weights.
    /// # Returns:
    /// - `AliasTable`: Table over `weights`.
    pub(crate) fn new(weights: &[f64]) -> Self {
        let n = weights.len();
        let total = weights.iter().sum::<f64>();
        let mut keep = weights
            .iter()
            .map(|w| w * n as f64 / total)
            .collect::<Vec<_>>();
        let mut alias = (0..n).collect::<Vec<_>>();

        // Pair every column below the mean with one above it, which donates the remainder.
        let (mut small, mut large): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| keep[i] < 1.0);
        while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
            small.pop();
            alias[s] = l;
            keep[l] -= 1.0 - keep[s];
            if keep[l] < 1.0 {
                large.pop();
                small.push(l);
            }
        }
        // Columns left over differ from one only by rounding.
        for i in small.into_iter().chain(large) {
            keep[i] = 1.0;
        }
        Self { keep, alias, total }
    }

    /// Sum of the weights.
    /// # Arguments:
    /// - `self`: Table.
    /// # Returns:
    /// - `f64`: Total weight, zero for an empty table.
    pub(crate) fn total(&self) -> f64 {
        self.total
    }

    /// Draw one index.
    /// # Arguments:
    /// - `self`: Nonempty table.
    /// - `rng`: Random-number generator.
    /// # Returns:
    /// - `usize`: Index `i` drawn with probability `w_i/\sum_j w_j`.
    #[inline(always)]
    pub(crate) fn sample(
        &self,
        rng: &mut QmcRng,
    ) -> usize {
        let n = self.keep.len();
        let u = rng.r#gen::<f64>() * n as f64;
        let i = (u as usize).min(n - 1);
        if u - (i as f64) < self.keep[i] {
            i
        } else {
            self.alias[i]
        }
    }
}
