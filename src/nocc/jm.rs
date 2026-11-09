// nocc/jm.rs
//! Jeziorski-Monkhorst layer shared by the deterministic and stochastic JM-NOCCMC propagations:
//! enumeration of each reference's determinants, the effective overlap and Hamiltonian, and the
//! state energies of the effective Hamiltonian.

// External crate imports.
use itertools::Itertools;
use ndarray::Array2;
use ndarray_linalg::{Eig, Inverse};
use num_complex::Complex64;

/// Effective overlap `S^e_{zx}=\langle{}^z\Psi|\Phi_x\rangle` and Hamiltonian
/// `H^e_{zx}=\langle{}^z\Psi|\hat H|\Phi_x\rangle` of the correlated references, evaluated
/// exactly or accumulated from samples. Stochastic samples are divided by the reference
/// populations `N_x` before accumulation, so the ratio estimate `G=(S^e)^{-1}H^e` is independent
/// of the population growth.
pub struct Effective {
    /// Effective overlap `S^e`.
    pub se: Array2<f64>,
    /// Effective Hamiltonian elements `H^e`.
    pub he: Array2<f64>,
}

impl Effective {
    /// Construct zero matrices for `M` references.
    /// # Arguments:
    /// - `m`: Number of references.
    /// # Returns:
    /// - `Effective`: Zero matrices.
    pub(in crate::nocc) fn new(m: usize) -> Self {
        Self {
            se: Array2::zeros((m, m)),
            he: Array2::zeros((m, m)),
        }
    }

    /// Add another accumulation.
    /// # Arguments:
    /// - `self`: Accumulators.
    /// - `other`: Accumulation to add.
    /// # Returns:
    /// - `()`: Adds `other` in place.
    pub(in crate::nocc) fn add(
        &mut self,
        other: &Effective,
    ) {
        self.se += &other.se;
        self.he += &other.he;
    }

    /// Scale the accumulation, as when forgetting old report blocks.
    /// # Arguments:
    /// - `self`: Accumulators.
    /// - `factor`: Scale factor.
    /// # Returns:
    /// - `()`: Scales both matrices in place.
    pub(in crate::nocc) fn scale(
        &mut self,
        factor: f64,
    ) {
        self.se *= factor;
        self.he *= factor;
    }

    /// Effective Hamiltonian `G=(S^e)^{-1}H^e`.
    /// # Arguments:
    /// - `self`: Effective overlap and Hamiltonian.
    /// # Returns:
    /// - `Option<Array2<f64>>`: `G`, or `None` if `S^e` is singular or `G` is not finite.
    pub(in crate::nocc) fn g(&self) -> Option<Array2<f64>> {
        let g = self.se.inv().ok()?.dot(&self.he);
        g.iter().all(|v| v.is_finite()).then_some(g)
    }
}

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
