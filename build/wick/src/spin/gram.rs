// spin/gram.rs
//! Spin Gram algebra of `S_k` on two-state spins.
//!
//! With the upper indices of a rank-`k` spin-orbital cumulant fixed, its lower indices may be
//! permuted by any `\pi \in S_k`. On spin-\tfrac12 labels these permutations have the Gram
//! matrix `G_{\pi\rho} = \mathrm{sgn}(\pi)\mathrm{sgn}(\rho)2^{c(\pi^{-1}\rho)}`, where `c`
//! counts cycles. `G` is singular for `k \ge 3`, reflecting the linear relations between
//! permuted spin-free cumulants.

// Standard library imports.
use std::sync::OnceLock;

// External crate imports.
use itertools::Itertools;
use num_rational::Ratio;

/// Permutations of `S_k` and their spin Gram matrix.
#[derive(Clone, Debug)]
pub(crate) struct Basis {
    /// Every permutation of `S_k`.
    pub(crate) ps: Vec<Vec<usize>>,
    /// `k!` by `k!` spin Gram matrix.
    pub(crate) g: Vec<Vec<Ratio<i64>>>,
}

impl Basis {
    /// Return the cached basis of rank `k`.
    /// # Arguments:
    /// - `k`: Permutation rank, `0..=4`.
    /// # Returns:
    /// - `&'static Self`: Permutations and Gram matrix.
    /// # Panics
    /// - Panics if `k > 4`.
    pub(crate) fn for_rank(k: usize) -> &'static Self {
        static BASES: OnceLock<[Basis; 5]> = OnceLock::new();

        &BASES.get_or_init(|| std::array::from_fn(Basis::new))[k]
    }

    /// Build the basis of rank `k`.
    /// # Arguments:
    /// - `k`: Permutation rank.
    /// # Returns:
    /// - `Self`: Permutations and Gram matrix.
    fn new(k: usize) -> Self {
        let ps = (0..k).permutations(k).collect::<Vec<_>>();
        let g = ps
            .iter()
            .map(|p| {
                ps.iter()
                    .map(|q| {
                        let r = permutation_compose(&permutation_inverse(p), q);
                        Ratio::from_integer(
                            permutation_sign(p)
                                * permutation_sign(q)
                                * (1i64 << permutation_cycles(&r)),
                        )
                    })
                    .collect()
            })
            .collect();

        Self { ps, g }
    }
}

/// Solve a consistent rational linear system `A x = b` by Gauss-Jordan elimination.
/// # Arguments:
/// - `a`: Matrix.
/// - `b`: Right-hand side.
/// # Returns:
/// - `Option<Vec<Ratio<i64>>>`: Solution with free variables set to zero, or `None` when the
///   system is inconsistent.
pub(crate) fn solve_rational_system(
    mut a: Vec<Vec<Ratio<i64>>>,
    b: Vec<Ratio<i64>>,
) -> Option<Vec<Ratio<i64>>> {
    let zero = Ratio::from_integer(0);
    let rows = a.len();
    let cols = a.first().map_or(0, Vec::len);
    for (row, x) in a.iter_mut().zip(b) {
        row.push(x);
    }

    // Reduce to row echelon form with unit pivots.
    let mut pivots = Vec::new();
    for col in 0..cols {
        let rank = pivots.len();
        let Some(p) = (rank..rows).find(|&r| a[r][col] != zero) else {
            continue;
        };
        a.swap(rank, p);

        let q = a[rank][col];
        for x in &mut a[rank][col..] {
            *x /= q;
        }
        let pivot = a[rank][col..].to_vec();
        for (r, row) in a.iter_mut().enumerate() {
            let q = row[col];
            if r == rank || q == zero {
                continue;
            }
            for (x, &y) in row[col..].iter_mut().zip(&pivot) {
                *x -= q * y;
            }
        }
        pivots.push(col);
    }

    // A zero row with a nonzero right-hand side is inconsistent.
    if a[pivots.len()..].iter().any(|row| row[cols] != zero) {
        return None;
    }

    let mut x = vec![zero; cols];
    for (r, &col) in pivots.iter().enumerate() {
        x[col] = a[r][cols];
    }

    Some(x)
}

/// Return the sign of a permutation, `\mathrm{sgn}(p) = (-1)^{N_{\text{inv}}}`.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `i64`: Sign.
pub(crate) fn permutation_sign(p: &[usize]) -> i64 {
    let n = (0..p.len())
        .tuple_combinations()
        .filter(|&(i, j)| p[i] > p[j])
        .count();

    if n % 2 == 0 { 1 } else { -1 }
}

/// Return the inverse permutation, `p^{-1}(p(i)) = i`.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `Vec<usize>`: Inverse permutation.
fn permutation_inverse(p: &[usize]) -> Vec<usize> {
    let mut out = vec![0; p.len()];
    for (i, &x) in p.iter().enumerate() {
        out[x] = i;
    }
    out
}

/// Compose two permutations, `(p \circ q)(i) = p(q(i))`.
/// # Arguments:
/// - `p`: Outer permutation.
/// - `q`: Inner permutation.
/// # Returns:
/// - `Vec<usize>`: Composition.
fn permutation_compose(
    p: &[usize],
    q: &[usize],
) -> Vec<usize> {
    q.iter().map(|&i| p[i]).collect()
}

/// Count the cycles of a permutation.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `usize`: Number of cycles.
fn permutation_cycles(p: &[usize]) -> usize {
    let mut seen = vec![false; p.len()];
    let mut out = 0;

    for i in 0..p.len() {
        if seen[i] {
            continue;
        }
        out += 1;
        let mut j = i;
        while !seen[j] {
            seen[j] = true;
            j = p[j];
        }
    }

    out
}
