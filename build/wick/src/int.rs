// int.rs

use crate::gram::{MAX, RMAX};
use num_rational::Ratio;

/// Test whether selected integer columns span one target.
/// # Arguments:
/// - `a`: Integer Gram columns.
/// - `b`: Scaled target vector.
/// - `s`: Candidate support columns.
/// # Returns:
/// - `Option<bool>`: Whether target lies in exact column span.
pub(crate) fn selected_columns_span_target(
    a: &[[i128; MAX]],
    b: &[i128],
    s: &[usize],
) -> Option<bool> {
    let rows = b.len();
    let cols = s.len();
    let mut m = [[0; RMAX + 1]; RMAX];

    for (i, row) in m.iter_mut().enumerate().take(rows) {
        for (j, &col) in s.iter().enumerate() {
            row[j] = a[col][i];
        }

        row[cols] = b[i];
    }

    columns_span_target(&mut m, rows, cols)
}

/// Test whether one integer augmented system is consistent.
/// # Arguments:
/// - `a`: Matrix scratch.
/// - `rows`: Row count.
/// - `cols`: Column count.
/// # Returns:
/// - `Option<bool>`: Whether the selected columns span the target.
fn columns_span_target(
    a: &mut [[i128; RMAX + 1]],
    rows: usize,
    cols: usize,
) -> Option<bool> {
    let mut r = 0;
    let mut d = 1;

    for c in 0..cols {
        let Some(piv) = (r..rows).find(|&i| a[i][c] != 0) else {
            continue;
        };

        a.swap(r, piv);
        let p = a[r][c];

        for i in 0..rows {
            if i == r || a[i][c] == 0 {
                continue;
            }

            for j in c + 1..=cols {
                let x = p.checked_mul(a[i][j])?;
                let y = a[i][c].checked_mul(a[r][j])?;
                a[i][j] = x.checked_sub(y)?.checked_div(d)?;
            }

            a[i][c] = 0;
        }

        d = p;
        r += 1;

        if r == rows {
            break;
        }
    }

    Some(
        !a.iter()
            .take(rows)
            .skip(r)
            .any(|row| row[..cols].iter().all(|&x| x == 0) && row[cols] != 0),
    )
}

/// Scale one rational vector to integer components.
/// # Arguments:
/// - `y`: Rational vector.
/// # Returns:
/// - `Option<Vec<i128>>`: Scaled integer vector if it fits.
pub(crate) fn scale_vector(y: &[Ratio<i64>]) -> Option<Vec<i128>> {
    let mut den = 1i128;

    for x in y {
        den = lowest_common_multiplier(den, *x.denom() as i128)?;
    }

    let mut out = Vec::with_capacity(y.len());

    for x in y {
        let q = den.checked_div(*x.denom() as i128)?;
        let z = (*x.numer() as i128).checked_mul(q)?;
        out.push(z);
    }

    Some(out)
}

/// Compute one least common multiple.
/// # Arguments:
/// - `a`: First integer.
/// - `b`: Second integer.
/// # Returns:
/// - `Option<i128>`: LCM if it fits.
fn lowest_common_multiplier(
    a: i128,
    b: i128,
) -> Option<i128> {
    let g = greatest_common_divisor(a, b);
    a.checked_div(g)?.checked_mul(b)
}

/// Compute one greatest common divisor.
/// # Arguments:
/// - `a`: First integer.
/// - `b`: Second integer.
/// # Returns:
/// - `i128`: GCD.
fn greatest_common_divisor(
    mut a: i128,
    mut b: i128,
) -> i128 {
    a = a.abs();
    b = b.abs();

    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }

    a
}
