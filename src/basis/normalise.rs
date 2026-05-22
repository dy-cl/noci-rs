// basis/normalise.rs

use ndarray::Array2;

use crate::HSCFState;
use crate::scf::normalise_hermitian;

/// Build the Hermitian-normalised h-SCF reference basis used for NOCI/SNOCI matrix elements.
/// The input states are not mutated, so generated h-SCF states remain in the holomorphic orbital convention required for branch tracking between geometries.
/// # Arguments
/// - `hstates`: Holomorphic h-SCF states generated for the current geometry.
/// - `s`: AO overlap matrix.
/// # Returns
/// - `Vec<HSCFState>`: NOCI-basis h-SCF states with Hermitian-normalised orbitals and compact parent indices.
pub fn hermitian_hnoci_basis(
    hstates: &[HSCFState],
    s: &Array2<f64>,
) -> Vec<HSCFState> {
    let mut out: Vec<HSCFState> = hstates.iter().filter(|st| st.noci_basis).cloned().collect();

    for (i, st) in out.iter_mut().enumerate() {
        normalise_hermitian(st, s);
        st.parent = i;
    }

    out
}
