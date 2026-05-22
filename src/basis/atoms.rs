// basis/atoms.rs

/// Compute the number of atoms represented by AO labels.
/// # Arguments
/// - `aolabels`: Labels which map AOs to atoms.
/// # Returns
/// - `usize`: Number of atoms represented in the AO labels.
pub(crate) fn atom_count(aolabels: &[String]) -> usize {
    aolabels
        .iter()
        .map(|s| {
            s.split_whitespace()
                .next()
                .unwrap()
                .parse::<usize>()
                .unwrap()
        })
        .max()
        .unwrap_or(0)
        + 1
}

/// Build AO-index lists for each atom represented by AO labels.
/// # Arguments
/// - `aolabels`: Labels which map AOs to atoms.
/// # Returns
/// - `Vec<Vec<usize>>`: AO indices grouped by atom.
pub(crate) fn atomao_for_labels(aolabels: &[String]) -> Vec<Vec<usize>> {
    (0..atom_count(aolabels))
        .map(|a| ao_indices_for_atomset(aolabels, &[a]))
        .collect()
}

/// Given AO labels containing atom indices, find the AO indices belonging to a set of atoms.
/// For example, if `aolabels = ["0 1s", "0 1s", "1 1s", "1 1s"]` and `atoms = [0]`, this returns `[0, 1]`.
/// # Arguments
/// `aolabels`: Labels which map AOs to atoms.
/// `atoms`: Atom indices for which we wish to know the corresponding AO indices.
/// # Returns
/// - `Vec<usize>`: AO indices belonging to the requested atom set.
pub(crate) fn ao_indices_for_atomset(
    aolabels: &[String],
    atoms: &[usize],
) -> Vec<usize> {
    // Iterate over all AO labels which contain for example "2 1s".
    aolabels
        .iter()
        .enumerate()
        // Take the first part of the label (e.g. "2") and keep it if the AOs atom index is in
        // atoms list.
        .filter(|(_, s)| {
            let a = s
                .split_whitespace()
                .next()
                .unwrap()
                .parse::<usize>()
                .unwrap();
            atoms.contains(&a)
            // Return the AO indices i.
        })
        .map(|(i, _)| i)
        .collect()
}
