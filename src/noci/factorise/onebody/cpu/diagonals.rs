// noci/factorise/onebody/cpu/diagonals.rs
//! CPU diagonal construction for factorised one-body NOCI operator contractions.

// Standard library imports.
use std::path::Path;

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::input::SNOCIStorage;
use crate::noci::fock::calculate_f_pair_orthogonal;
use crate::noci::overlap::calculate_s_pair_orthogonal;
use crate::noci::types::{FockData, FockMOCache, NOCIData, NOCIScalar};

// Parent/sibling imports.
use super::super::super::storage::OneBodyStoragePlan;
use super::super::super::{ParentSpinSpace, SpinFactorisation};
use super::backend::OneBodyBlock;
use super::factors::{FactorisedOneBodyBlock, build_one_body_factor_tables};

/// Build diagonal entries of `F + \lambda S` and `S` from cached same-spin factors.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `blocks`: Cached parent-pair blocks indexed as `Q * nparent + P`.
/// - `nparent`: Number of parent references.
/// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
/// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
/// - `lambda`: Scalar overlap shift in `F + \lambda S`.
/// # Returns
/// - `(Array1<T>, Array1<T>)`: Diagonal of `F + \lambda S` and diagonal of `S`.
pub(super) fn one_body_diagonals<T: NOCIScalar>(
    spin: &SpinFactorisation,
    blocks: &[OneBodyBlock<T>],
    nparent: usize,
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    lambda: T,
) -> (Array1<T>, Array1<T>) {
    let zero = T::from_real(0.0);
    let ndet = spin.aids.len();
    let mut m_diag = vec![zero; ndet];
    let mut s_diag = vec![zero; ndet];

    for (parent_id, parent) in spin.parents.iter().enumerate() {
        if parent.entries.is_empty() {
            continue;
        }
        match &blocks[parent_id * nparent + parent_id] {
            OneBodyBlock::Orthogonal(_) => {
                fill_orthogonal_one_body_diagonal_block(
                    parent,
                    data,
                    &fock.fock_mocache[parent_id],
                    lambda,
                    &mut m_diag,
                    &mut s_diag,
                );
            }
            OneBodyBlock::Factorised(block) => {
                fill_one_body_diagonal_block(parent, block, lambda, &mut m_diag, &mut s_diag);
            }
            OneBodyBlock::Transient(block) => {
                let mut storage_plan =
                    OneBodyStoragePlan::new(Path::new("."), 0, 0, SNOCIStorage::RAM);
                let block = build_one_body_factor_tables(
                    spin,
                    data,
                    &mut storage_plan,
                    block.target_parent,
                    block.source_parent,
                    block.lp,
                    block.gp,
                    block.target_left,
                    spin.parents[block.target_parent].areps.len(),
                    spin.parents[block.target_parent].breps.len(),
                    spin.parents[block.source_parent].areps.len(),
                    spin.parents[block.source_parent].breps.len(),
                    block.contraction,
                );
                fill_one_body_diagonal_block(parent, &block, lambda, &mut m_diag, &mut s_diag);
            }
        }
    }

    (Array1::from_vec(m_diag), Array1::from_vec(s_diag))
}

/// Fill determinant diagonals from one same-parent factor block.
/// # Arguments:
/// - `parent`: Parent-local actual determinant entries.
/// - `block`: Same-parent factor block.
/// - `lambda`: Scalar overlap shift in `F + \lambda S`.
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
pub(super) fn fill_one_body_diagonal_block<T: NOCIScalar>(
    parent: &ParentSpinSpace,
    block: &FactorisedOneBodyBlock<T>,
    lambda: T,
    m_diag: &mut [T],
    s_diag: &mut [T],
) {
    let (sa, fa, sb, fb) = block.factors.factors();
    for entry in &parent.entries {
        let saa = sa[entry.a * block.nsa + entry.a];
        let faa = fa[entry.a * block.nsa + entry.a];
        let sbb = sb[entry.b * block.nsb + entry.b];
        let fbb = fb[entry.b * block.nsb + entry.b];
        let s = saa * sbb;
        s_diag[entry.det] = s;
        m_diag[entry.det] = faa * sbb + saa * fbb + lambda * s;
    }
}

/// Fill same-parent orthogonal diagonals from parent-local Slater-Condon rules.
/// # Arguments:
/// - `parent`: Parent-local actual determinant entries.
/// - `data`: Shared NOCI determinant data.
/// - `cache`: MO-basis Fock cache for the parent.
/// - `lambda`: Scalar overlap shift in `F + \lambda S`.
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
pub(super) fn fill_orthogonal_one_body_diagonal_block<T: NOCIScalar>(
    parent: &ParentSpinSpace,
    data: &NOCIData<'_, T>,
    cache: &FockMOCache<T>,
    lambda: T,
    m_diag: &mut [T],
    s_diag: &mut [T],
) {
    for entry in &parent.entries {
        let det = &data.basis[entry.det];
        let s = calculate_s_pair_orthogonal(det, det);
        s_diag[entry.det] = s;
        m_diag[entry.det] = calculate_f_pair_orthogonal(cache, det, det) + lambda * s;
    }
}
