// scf/select.rs

use ndarray::{Array1, Array2, Axis};

/// Select occupied MO indices using Aufbau ordering.
/// # Arguments
/// - `e`: MO energies.
/// - `nocc`: Number of occupied MOs.
/// # Returns
/// - `Vec<usize>`: Occupied MO indices.
pub(crate) fn aufbau_indices(e: &Array1<f64>, nocc: usize) -> Vec<usize> {
    let mut idx: Vec<_> = (0..e.len()).collect();
    idx.sort_by(|&i, &j| e[i].partial_cmp(&e[j]).unwrap());
    idx.truncate(nocc);
    idx
}

/// Select occupied MO column indices by the maximum overlap method.
/// Forms `O = C_old^T S C` and scores each current MO by its overlap with the previous occupied space.
/// # Arguments
/// - `c_occ_old`: Previous occupied MO coefficients.
/// - `c`: Current MO coefficients.
/// - `s`: AO overlap matrix.
/// - `nocc`: Number of occupied spin orbitals to select.
/// # Returns
/// - `Vec<usize>`: Occupied MO indices selected by MOM.
pub(crate) fn mom_select(c_occ_old: &Array2<f64>, c: &Array2<f64>, s: &Array2<f64>, nocc: usize) -> Vec<usize> {
    let o = c_occ_old.t().dot(s).dot(c);
    let p = o.mapv(|x| x.abs()).sum_axis(Axis(0));
    let mut idx: Vec<_> = (0..p.len()).collect();
    idx.sort_by(|&i, &j| p[j].partial_cmp(&p[i]).unwrap());
    idx.truncate(nocc);
    idx
}
