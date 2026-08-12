// noci/factorise/onebody/gpu/diagonals.rs
//! GPU diagonal construction for factorised one-body NOCI operator contractions.

// External crate imports.
use cubecl::prelude::*;
use ndarray::Array1;

// Crate-root imports.
use crate::noci::types::NOCIScalar;

/// Build diagonal entries of `F + \lambda S` and `S` from device factors.
/// # Arguments:
/// - `n`: Number of determinant-space diagonal entries.
/// - `lambda`: Scalar overlap shift in `F + \lambda S`.
/// # Returns
/// - `(Array1<T>, Array1<T>)`: Diagonal of `F + \lambda S` and diagonal of `S`.
pub(crate) fn one_body_diagonals<T: NOCIScalar>(
    _n: usize,
    _lambda: T,
) -> (Array1<T>, Array1<T>) {
    eprintln!("GPU one-body diagonal construction is not implemented yet");
    std::process::exit(1);
}

/// Fill determinant diagonals from one same-parent factor block on the device.
/// # Arguments:
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
pub(crate) fn fill_one_body_diagonal_block<T: NOCIScalar>(
    _m_diag: &mut Array1<T>,
    _s_diag: &mut Array1<T>,
) {
    eprintln!("GPU factorised one-body diagonal block construction is not implemented yet");
    std::process::exit(1);
}

/// Fill same-parent orthogonal diagonals from parent-local Slater-Condon rules on the device.
/// # Arguments:
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
pub(crate) fn fill_orthogonal_one_body_diagonal_block<T: NOCIScalar>(
    _m_diag: &mut Array1<T>,
    _s_diag: &mut Array1<T>,
) {
    eprintln!("GPU orthogonal one-body diagonal block construction is not implemented yet");
    std::process::exit(1);
}

/// Fill determinant diagonals from one same-parent factor block on the device.
/// # Arguments:
/// - `sa`: Row-major alpha overlap factors.
/// - `fa`: Row-major alpha Fock factors.
/// - `sb`: Row-major beta overlap factors.
/// - `fb`: Row-major beta Fock factors.
/// - `entry_det`: Parent entry determinant IDs.
/// - `entry_a`: Parent entry alpha components.
/// - `entry_b`: Parent entry beta components.
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// - `lambda`: Scalar overlap shift.
/// - `nentry`: Number of parent entries.
/// - `nsa`: Source alpha component count.
/// - `nsb`: Source beta component count.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
#[cube(launch_unchecked)]
pub(crate) fn fill_one_body_diagonal_block_kernel(
    sa: &Array<f64>,
    fa: &Array<f64>,
    sb: &Array<f64>,
    fb: &Array<f64>,
    entry_det: &Array<u32>,
    entry_a: &Array<u32>,
    entry_b: &Array<u32>,
    m_diag: &mut Array<f64>,
    s_diag: &mut Array<f64>,
    lambda: f64,
    nentry: u32,
    nsa: u32,
    nsb: u32,
) {
    if ABSOLUTE_POS >= nentry {
        return;
    }
    let a = entry_a[ABSOLUTE_POS];
    let b = entry_b[ABSOLUTE_POS];
    let saa = sa[a * nsa + a];
    let faa = fa[a * nsa + a];
    let sbb = sb[b * nsb + b];
    let fbb = fb[b * nsb + b];
    let s = saa * sbb;
    let det = entry_det[ABSOLUTE_POS];
    s_diag[det] = s;
    m_diag[det] = faa * sbb + saa * fbb + lambda * s;
}
