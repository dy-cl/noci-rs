// spinsum.rs

use itertools::Itertools;
use num_rational::Ratio;
use num_traits::Zero;

pub type Rat = Ratio<i64>;

/// Return all permutations of `0..n`.
/// # Arguments:
/// - `n`: Number of elements.
/// # Returns:
/// - `Vec<Vec<usize>>`: Permutations.
pub fn perms(n: usize) -> Vec<Vec<usize>> {
    (0..n).permutations(n).collect()
}

/// Build the spin Gram matrix.
/// # Arguments:
/// - `n`: Cumulant rank.
/// # Returns:
/// - `Vec<Vec<Rat>>`: Gram matrix.
pub fn gram(n: usize) -> Vec<Vec<Rat>> {
    let ps = perms(n);

    ps.iter()
        .map(|p| {
            ps.iter()
                .map(|q| {
                    let r = comp(&inv(p), q);
                    Rat::from_integer(sgn(p) * sgn(q) * (1_i64 << cyc(&r)))
                })
                .collect()
        })
        .collect()
}

/// Solve a rational linear system if consistent.
/// # Arguments:
/// - `a`: Matrix.
/// - `b`: Right-hand side.
/// # Returns:
/// - `Option<Vec<Rat>>`: Solution with free variables set to zero.
pub fn solve(mut a: Vec<Vec<Rat>>, b: Vec<Rat>) -> Option<Vec<Rat>> {
    let rows = a.len();

    if rows == 0 {
        return Some(Vec::new());
    }

    let cols = a[0].len();

    for (r, x) in b.into_iter().enumerate() {
        a[r].push(x);
    }

    let mut piv = Vec::new();
    let mut row = 0;

    for col in 0..cols {
        let Some(p) = (row..rows).find(|&r| !a[r][col].is_zero()) else {
            continue;
        };

        a.swap(row, p);

        let q = a[row][col].clone();
        for j in col..=cols {
            a[row][j] /= q.clone();
        }

        for r in 0..rows {
            if r == row {
                continue;
            }

            let q = a[r][col].clone();

            if q.is_zero() {
                continue;
            }

            for j in col..=cols {
                let x = q.clone() * a[row][j].clone();
                a[r][j] -= x;
            }
        }

        piv.push(col);
        row += 1;

        if row == rows {
            break;
        }
    }

    for r in row..rows {
        if (0..cols).all(|c| a[r][c].is_zero()) && !a[r][cols].is_zero() {
            return None;
        }
    }

    let mut x = vec![Rat::zero(); cols];

    for (r, col) in piv.into_iter().enumerate() {
        x[col] = a[r][cols].clone();
    }

    Some(x)
}

/// Return permutation sign.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `i64`: Sign.
pub fn sgn(p: &[usize]) -> i64 {
    let n = (0..p.len()).cartesian_product(0..p.len())
        .filter(|&(i, j)| i < j && p[i] > p[j])
        .count();

    if n % 2 == 0 { 1 } else { -1 }
}

/// Return inverse permutation.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `Vec<usize>`: Inverse permutation.
pub fn inv(p: &[usize]) -> Vec<usize> {
    let mut out = vec![0; p.len()];

    for (i, &x) in p.iter().enumerate() {
        out[x] = i;
    }

    out
}

/// Compose two permutations.
/// # Arguments:
/// - `p`: Outer permutation.
/// - `q`: Inner permutation.
/// # Returns:
/// - `Vec<usize>`: Composition.
pub fn comp(p: &[usize], q: &[usize]) -> Vec<usize> {
    (0..p.len()).map(|i| p[q[i]]).collect()
}

/// Count cycles in a permutation.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `usize`: Number of cycles.
pub fn cyc(p: &[usize]) -> usize {
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
