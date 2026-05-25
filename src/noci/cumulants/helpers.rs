// noci/cumulants/helpers.rs

/// Generate set partitions of the slots `0..n`.
/// # Arguments:
/// - `n`: Number of slots.
/// # Returns:
/// - `Vec<Vec<Vec<usize>>>`: Unique set partitions.
pub(super) fn set_partitions(n: usize) -> Vec<Vec<Vec<usize>>> {
    fn rec(i: usize, n: usize, current: &mut Vec<Vec<usize>>, out: &mut Vec<Vec<Vec<usize>>>) {
        if i == n {
            out.push(current.clone());
            return;
        }

        for b in 0..current.len() {
            current[b].push(i);
            rec(i + 1, n, current, out);
            current[b].pop();
        }

        current.push(vec![i]);
        rec(i + 1, n, current, out);
        current.pop();
    }

    let mut out = Vec::new();
    rec(0, n, &mut Vec::new(), &mut out);
    out
}

/// Return the sign of a permutation.
/// # Arguments:
/// - `permutation`: Permutation mapping upper slots to lower slots.
/// # Returns:
/// - `i32`: `+1` for even permutations and `-1` for odd permutations.
pub(super) fn permutation_sign(permutation: &[usize]) -> i32 {
    let mut inversions = 0;

    for i in 0..permutation.len() {
        for j in (i + 1)..permutation.len() {
            if permutation[i] > permutation[j] {
                inversions += 1;
            }
        }
    }

    if inversions % 2 == 0 { 1 } else { -1 }
}

/// Count cycles in a permutation.
/// # Arguments:
/// - `permutation`: Permutation mapping upper slots to lower slots.
/// # Returns:
/// - `usize`: Number of permutation cycles.
pub(super) fn permutation_cycles(permutation: &[usize]) -> usize {
    let mut seen = vec![false; permutation.len()];
    let mut cycles = 0;

    for i in 0..permutation.len() {
        if seen[i] {
            continue;
        }

        cycles += 1;
        let mut j = i;

        while !seen[j] {
            seen[j] = true;
            j = permutation[j];
        }
    }

    cycles
}

/// Decode a flat tensor index into base-`n` indices.
/// # Arguments:
/// - `index`: Flat tensor index.
/// - `rank`: Number of decoded slots.
/// - `n`: Extent of each slot.
/// # Returns:
/// - `Vec<usize>`: Decoded tensor indices.
pub(super) fn decode_index(mut index: usize, rank: usize, n: usize) -> Vec<usize> {
    let mut out = vec![0; rank];

    for i in (0..rank).rev() {
        out[i] = index % n;
        index /= n;
    }

    out
}
