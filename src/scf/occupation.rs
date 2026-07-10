// scf/occupation.rs

use std::sync::Arc;

use ndarray::{Array1, Array2, Axis, s};

use crate::maths::adjoint;
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

/// Extract occupied and virtual orbital indices from one occupation bitstring.
/// # Arguments:
/// - `nmo`: Number of molecular orbitals.
/// - `occ`: Occupation bitstring.
/// # Returns
/// - `(Vec<usize>, Vec<usize>)`: Occupied and virtual orbital indices.
fn orbital_occupation(
    nmo: usize,
    occ: u128,
) -> (Vec<usize>, Vec<usize>) {
    let occupied = (0..nmo).filter(|&p| ((occ >> p) & 1u128) == 1).collect();

    let virtuals = (0..nmo).filter(|&p| ((occ >> p) & 1u128) == 0).collect();

    (occupied, virtuals)
}

/// Reorder one determinant state's MO columns so occupied orbitals precede virtual orbitals.
/// Occupied and virtual orbitals each preserve their original relative order.
/// # Arguments:
/// - `st`: Determinant state to clone into occupied-first orbital order.
/// # Returns
/// - `DetState<T>`: Clone of `st` in occupied-first orbital order.
pub fn occ_first<T: StateScalar>(st: &DetState<T>) -> DetState<T> {
    let (aocc, avirt) = orbital_occupation(st.ca.ncols(), st.oa);
    let naocc = aocc.len();

    let aorder: Vec<usize> = aocc.into_iter().chain(avirt).collect();

    let ca = st.ca.select(Axis(1), &aorder);
    let ca_occ = ca.slice(s![.., 0..naocc]).to_owned();

    let da = ca_occ.dot(&adjoint(&ca_occ));

    let oa = if naocc == 128 {
        u128::MAX
    } else {
        (1u128 << naocc) - 1
    };

    let (bocc, bvirt) = orbital_occupation(st.cb.ncols(), st.ob);
    let nbocc = bocc.len();

    let border: Vec<usize> = bocc.into_iter().chain(bvirt).collect();

    let cb = st.cb.select(Axis(1), &border);
    let cb_occ = cb.slice(s![.., 0..nbocc]).to_owned();

    let db = cb_occ.dot(&adjoint(&cb_occ));

    let ob = if nbocc == 128 {
        u128::MAX
    } else {
        (1u128 << nbocc) - 1
    };

    let mut out = st.clone();

    out.ca = Arc::new(ca);
    out.cb = Arc::new(cb);
    out.da = Arc::new(da);
    out.db = Arc::new(db);

    out.oa = oa;
    out.ob = ob;

    out
}

/// Extract occupied and virtual orbital indices from determinant occupation bitstrings.
/// # Arguments:
/// - `st`: Determinant state whose orbital occupations are being inspected.
/// # Returns
/// - `SpinOccupation`: Occupied and virtual MO indices for alpha and beta spin.
pub fn spin_occupation<T: StateScalar>(st: &DetState<T>) -> SpinOccupation {
    let (occ_alpha, virt_alpha) = orbital_occupation(st.ca.ncols(), st.oa);

    let (occ_beta, virt_beta) = orbital_occupation(st.cb.ncols(), st.ob);

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
