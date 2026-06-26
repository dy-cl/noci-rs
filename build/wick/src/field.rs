// field.rs

use crate::gram::{MAX, RMAX};
use num_rational::Ratio;
use std::sync::OnceLock;

/// Build certificate modular Gram projections.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// - `rows`: Independent row indices.
/// # Returns:
/// - `[Vec<[u64; MAX]>; 3]`: Gram columns modulo certificate primes.
pub(crate) fn columns_u64(
    g: &[Vec<Ratio<i64>>],
    rows: &[usize],
) -> [Vec<[u64; MAX]>; 3] {
    [
        column_u64(g, rows, 1000000007),
        column_u64(g, rows, 1000000009),
        column_u64(g, rows, 998244353),
    ]
}

/// Build one certificate modular Gram projection.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// - `rows`: Independent row indices.
/// - `p`: Prime modulus.
/// # Returns:
/// - `Vec<[u64; MAX]>`: Gram columns modulo `p`.
fn column_u64(
    g: &[Vec<Ratio<i64>>],
    rows: &[usize],
    p: u64,
) -> Vec<[u64; MAX]> {
    let n = g.len();
    let mut a = vec![[0; MAX]; n];

    for (i, &row) in rows.iter().enumerate() {
        for (j, x) in g[row].iter().enumerate() {
            a[j][i] = modulo_i128_u64(*x.numer() as i128, p);
        }
    }

    a
}

/// Build fast modular Gram projection.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// - `rows`: Independent row indices.
/// # Returns:
/// - `Vec<[u16; MAX]>`: Gram columns modulo 65521.
pub(crate) fn columns_fast_u16(
    g: &[Vec<Ratio<i64>>],
    rows: &[usize],
) -> Vec<[u16; MAX]> {
    let n = g.len();
    let mut a = vec![[0; MAX]; n];

    for (i, &row) in rows.iter().enumerate() {
        for (j, x) in g[row].iter().enumerate() {
            a[j][i] = modulo_i128_u64(*x.numer() as i128, 65521) as u16;
        }
    }

    a
}

/// Build certificate modular target vectors.
/// # Arguments:
/// - `b`: Scaled integer target.
/// # Returns:
/// - `[[u64; MAX]; 3]`: Target vectors modulo certificate primes.
pub(crate) fn vectors_u64(b: &[i128]) -> [[u64; MAX]; 3] {
    [
        vector_u64(b, 1000000007),
        vector_u64(b, 1000000009),
        vector_u64(b, 998244353),
    ]
}

/// Build one certificate modular target vector.
/// # Arguments:
/// - `b`: Scaled integer target.
/// - `p`: Prime modulus.
/// # Returns:
/// - `[u64; MAX]`: Target vector modulo `p`.
fn vector_u64(
    b: &[i128],
    p: u64,
) -> [u64; MAX] {
    let mut y = [0; MAX];

    for (i, &x) in b.iter().enumerate() {
        y[i] = modulo_i128_u64(x, p);
    }

    y
}

/// Build fast modular target vector.
/// # Arguments:
/// - `b`: Scaled integer target.
/// # Returns:
/// - `[u16; MAX]`: Target vector modulo 65521.
pub(crate) fn vector_fast_u16(b: &[i128]) -> [u16; MAX] {
    let mut y = [0; MAX];

    for (i, &x) in b.iter().enumerate() {
        y[i] = modulo_i128_u64(x, 65521) as u16;
    }

    y
}

/// Project one integer into one prime field.
/// # Arguments:
/// - `x`: Integer value.
/// - `p`: Prime modulus.
/// # Returns:
/// - `u64`: Modular value.
fn modulo_i128_u64(
    x: i128,
    p: u64,
) -> u64 {
    if x >= 0 {
        (x as u128 % p as u128) as u64
    } else {
        let y = (x.unsigned_abs() % p as u128) as u64;
        if y == 0 { 0 } else { p - y }
    }
}

/// Test all certificate modular span filters.
/// # Arguments:
/// - `n`: Reduced row count.
/// - `f`: Modular Gram projections.
/// - `b`: Modular target projections.
/// - `s`: Candidate support columns.
/// # Returns:
/// - `bool`: False only when exact span is impossible.
pub(crate) fn spans_u64(
    n: usize,
    f: &[Vec<[u64; MAX]>; 3],
    b: &[[u64; MAX]; 3],
    s: &[usize],
) -> bool {
    spans_one_u64(1000000007, n, &f[0], &b[0], s)
        && spans_one_u64(1000000009, n, &f[1], &b[1], s)
        && spans_one_u64(998244353, n, &f[2], &b[2], s)
}

/// Test one certificate modular span filter.
/// # Arguments:
/// - `p`: Prime modulus.
/// - `n`: Reduced row count.
/// - `f`: Prime-field projection.
/// - `b`: Prime-field target.
/// - `s`: Candidate support columns.
/// # Returns:
/// - `bool`: False only when exact span is impossible.
fn spans_one_u64(
    p: u64,
    n: usize,
    f: &[[u64; MAX]],
    b: &[u64; MAX],
    s: &[usize],
) -> bool {
    let cols = s.len();
    let mut a = [[0; RMAX + 1]; RMAX];

    for i in 0..n {
        for (j, &col) in s.iter().enumerate() {
            a[i][j] = f[col][i];
        }

        a[i][cols] = b[i];
    }

    consistent_u64(&mut a, n, cols, p)
}

/// Test one modular augmented system.
/// # Arguments:
/// - `a`: Matrix scratch.
/// - `rows`: Row count.
/// - `cols`: Column count.
/// - `p`: Prime modulus.
/// # Returns:
/// - `bool`: False only when a full-rank selected support cannot span target.
fn consistent_u64(
    a: &mut [[u64; RMAX + 1]; RMAX],
    rows: usize,
    cols: usize,
    p: u64,
) -> bool {
    let mut r = 0;

    for c in 0..cols {
        let Some(piv) = (r..rows).find(|&i| a[i][c] != 0) else {
            continue;
        };

        a.swap(r, piv);

        let q = invert_u64(a[r][c], p).unwrap();
        for j in c..=cols {
            a[r][j] = a[r][j] * q % p;
        }

        for i in 0..rows {
            if i == r || a[i][c] == 0 {
                continue;
            }

            let q = a[i][c];
            for j in c..=cols {
                let x = q * a[r][j] % p;
                a[i][j] = (a[i][j] + p - x) % p;
            }
        }

        r += 1;
        if r == rows {
            break;
        }
    }

    if r != cols {
        return true;
    }

    !a.iter()
        .take(rows)
        .skip(r)
        .any(|row| row[..cols].iter().all(|&x| x == 0) && row[cols] != 0)
}

/// Invert one modular value.
/// # Arguments:
/// - `x`: Modular value.
/// - `p`: Prime modulus.
/// # Returns:
/// - `Option<u64>`: Multiplicative inverse if nonzero.
fn invert_u64(
    x: u64,
    p: u64,
) -> Option<u64> {
    if x == 0 {
        None
    } else {
        let mut x = x;
        let mut n = p - 2;
        let mut out = 1;

        while n > 0 {
            if n & 1 == 1 {
                out = out * x % p;
            }

            x = x * x % p;
            n >>= 1;
        }

        Some(out)
    }
}

/// Invert one fast modular value.
/// # Arguments:
/// - `x`: Nonzero modular value.
/// # Returns:
/// - `u16`: Multiplicative inverse modulo 65521.
pub(crate) fn invert_fast_u16(x: u16) -> u16 {
    inverses_fast_u16()[x as usize]
}

/// Multiply two fast modular values.
/// # Arguments:
/// - `a`: First factor.
/// - `b`: Second factor.
/// # Returns:
/// - `u16`: Product modulo 65521.
pub(crate) fn multiply_fast_u16(
    a: u16,
    b: u16,
) -> u16 {
    let x = a as u32 * b as u32;
    let x = (x & 0xffff) + 15 * (x >> 16);
    let x = (x & 0xffff) + 15 * (x >> 16);

    if x >= 65521 {
        (x - 65521) as u16
    } else {
        x as u16
    }
}

/// Subtract two fast modular values.
/// # Arguments:
/// - `a`: Left value.
/// - `b`: Right value.
/// # Returns:
/// - `u16`: Difference modulo 65521.
pub(crate) fn subtract_fast_u16(
    a: u16,
    b: u16,
) -> u16 {
    let x = a as u32 + 65521 - b as u32;

    if x >= 65521 {
        (x - 65521) as u16
    } else {
        x as u16
    }
}

/// Return the fast modular inverse table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `&'static [u16; 65521]`: Multiplicative inverse table.
fn inverses_fast_u16() -> &'static [u16; 65521] {
    static INVS: OnceLock<[u16; 65521]> = OnceLock::new();

    INVS.get_or_init(|| {
        let mut out = [0; 65521];
        out[1] = 1;

        for i in 2..65521 {
            let x = i as u64;
            out[i] = (65521 - (65521 / x) * out[(65521 % x) as usize] as u64 % 65521) as u16;
        }

        out
    })
}
