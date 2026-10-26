// nocc/jm.rs
//! Jeziorski-Monkhorst layer shared by the deterministic and stochastic JM-NOCCMC propagations:
//! enumeration of each reference's determinants and the state energies of the effective
//! Hamiltonian.

// External crate imports.
use itertools::Itertools;
use ndarray::Array2;
use ndarray_linalg::Eig;
use num_complex::Complex64;

/// Enumerate the determinants of one reference up to an excitation rank, ordered by rank.
/// # Arguments:
/// - `oa`: Alpha occupation of the reference.
/// - `ob`: Beta occupation of the reference.
/// - `norb`: Number of spatial orbitals in the reference frame.
/// - `rank`: Highest excitation rank, capped at the largest possible rank.
/// # Returns:
/// - `Vec<(u128, u128, usize)>`: Alpha and beta occupations and excitation rank.
pub(in crate::nocc) fn reference_determinants(
    oa: u128,
    ob: u128,
    norb: usize,
    rank: usize,
) -> Vec<(u128, u128, usize)> {
    let split = |occ: u128| {
        let (occupied, virtuals): (Vec<usize>, Vec<usize>) =
            (0..norb).partition(|&p| occ & (1u128 << p) != 0);
        (occupied, virtuals)
    };
    let (aocc, avir) = split(oa);
    let (bocc, bvir) = split(ob);
    let excite = |occ: u128, holes: &[usize], parts: &[usize]| {
        let occ = holes.iter().fold(occ, |o, &h| o & !(1u128 << h));
        parts.iter().fold(occ, |o, &p| o | (1u128 << p))
    };

    let mut out = Vec::new();
    for order in 0..=rank {
        for arank in 0..=order {
            let brank = order - arank;
            for aholes in aocc.iter().copied().combinations(arank) {
                for aparts in avir.iter().copied().combinations(arank) {
                    for bholes in bocc.iter().copied().combinations(brank) {
                        for bparts in bvir.iter().copied().combinations(brank) {
                            out.push((
                                excite(oa, &aholes, &aparts),
                                excite(ob, &bholes, &bparts),
                                order,
                            ));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Eigenvalues of the effective Hamiltonian, sorted by real part.
/// # Arguments:
/// - `g`: Effective Hamiltonian.
/// # Returns:
/// - `Vec<Complex64>`: State energies.
pub(in crate::nocc) fn jm_energies(g: &Array2<f64>) -> Vec<Complex64> {
    let (w, _) = g.eig().unwrap();
    let mut w = w.to_vec();
    w.sort_by(|a, b| a.re.total_cmp(&b.re));
    w
}
