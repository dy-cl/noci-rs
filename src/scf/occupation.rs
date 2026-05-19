// scf/occupation.rs

use ndarray::{Array1, Array2};

use crate::{DetState, StateScalar};

pub struct SpinOccupation {
    /// Indices of occupied alpha-spin orbitals.
    pub occ_alpha: Vec<usize>,
    /// Indices of virtual alpha-spin orbitals.
    pub virt_alpha: Vec<usize>,
    /// Indices of occupied beta-spin orbitals.
    pub occ_beta: Vec<usize>,
    /// Indices of virtual beta-spin orbitals.
    pub virt_beta: Vec<usize>,
}

impl SpinOccupation {
    /// Return alpha-spin indices ordered as occupied followed by virtual.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Vec<usize>`: Alpha-spin occupied indices followed by virtual indices.
    pub(crate) fn alpha_occupied_first(&self) -> Vec<usize> {
        self.occ_alpha
            .iter()
            .chain(self.virt_alpha.iter())
            .copied()
            .collect()
    }

    /// Return beta-spin indices ordered as occupied followed by virtual.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Vec<usize>`: Beta-spin occupied indices followed by virtual indices.
    pub(crate) fn beta_occupied_first(&self) -> Vec<usize> {
        self.occ_beta
            .iter()
            .chain(self.virt_beta.iter())
            .copied()
            .collect()
    }
}

/// Extract occupied and virtual orbital indices from determinant occupation bitstrings.
/// # Arguments:
/// - `st`: Determinant state whose orbital occupations are being inspected.
/// # Returns:
/// - `SpinOccupation`: Occupied and virtual MO indices for alpha and beta spin.
pub fn spin_occupation<T: StateScalar>(st: &DetState<T>) -> SpinOccupation {
    let occ_alpha: Vec<usize> = (0..st.ca.ncols())
        .filter(|&p| ((st.oa >> p) & 1u128) == 1)
        .collect();
    let occ_beta: Vec<usize> = (0..st.cb.ncols())
        .filter(|&p| ((st.ob >> p) & 1u128) == 1)
        .collect();

    let virt_alpha: Vec<usize> = (0..st.ca.ncols())
        .filter(|&p| ((st.oa >> p) & 1u128) == 0)
        .collect();
    let virt_beta: Vec<usize> = (0..st.cb.ncols())
        .filter(|&p| ((st.ob >> p) & 1u128) == 0)
        .collect();

    SpinOccupation {
        occ_alpha,
        virt_alpha,
        occ_beta,
        virt_beta,
    }
}

/// Calculate MO occupancies as the diagonal of a density matrix in the MO basis.
/// `T = C^T S D S C`.
/// # Arguments
/// - `c`: Spin MO coefficient matrix.
/// - `d`: Spin density matrix.
/// - `s`: AO overlap matrix.
/// # Returns
/// - `Array1<f64>`: MO occupancies thresholded to zero or one.
pub(crate) fn mo_occupancies(
    c: &Array2<f64>,
    d: &Array2<f64>,
    s: &Array2<f64>,
) -> Array1<f64> {
    let t = c.t().dot(s).dot(d).dot(s).dot(c);
    let diag = t.diag().to_owned();
    diag.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 })
}

/// Convert occupied MO indices to a bitstring.
/// # Arguments
/// - `idx`: Occupied MO indices.
/// # Returns
/// - `u128`: Occupation bitstring.
pub(crate) fn occvec_to_bits(idx: &[usize]) -> u128 {
    let mut bits = 0u128;
    for &i in idx {
        bits |= 1u128 << i;
    }
    bits
}
