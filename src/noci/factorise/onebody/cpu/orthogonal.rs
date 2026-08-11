// noci/factorise/onebody/cpu/orthogonal.rs
//! CPU same-parent orthogonal one-body NOCI operator contractions.

// Standard library imports.
use std::collections::HashMap;

// Crate-root imports.
use crate::noci::fock::calculate_f_pair_orthogonal;
use crate::noci::overlap::calculate_s_pair_orthogonal;
use crate::noci::types::{FockMOCache, NOCIData, NOCIScalar};

// Parent/sibling imports.
use super::super::super::SpinFactorisation;

/// Actual determinant target in one orthogonal occupation group.
#[derive(Clone, Copy)]
struct OrthogonalTarget {
    /// Global determinant index `I`.
    det: usize,
    /// Parent-local alpha component `a_I`.
    a: usize,
}

/// Parent-local occupation group for the orthogonal one-body shortcut.
struct OrthogonalOccupationGroup {
    /// Determinants sharing one occupation pair.
    targets: Vec<OrthogonalTarget>,
}

/// Parent-local occupation lookup for same-parent orthogonal one-body application.
pub(super) struct OrthogonalOneBodyBlock {
    /// Parent `P`.
    pub(super) parent: usize,
    /// Occupation-pair ID keyed by `(o_alpha,o_beta)`.
    opos: HashMap<(u128, u128), usize>,
    /// Determinants grouped by occupation pair.
    groups: Vec<OrthogonalOccupationGroup>,
}

/// Shared data for same-parent orthogonal one-body application.
pub(super) struct OrthogonalApplyContext<'a, T: NOCIScalar> {
    /// Shared NOCI data used to read determinant states.
    pub(super) data: &'a NOCIData<'a, T>,
    /// MO-basis Fock cache for the parent.
    pub(super) cache: &'a FockMOCache<T>,
    /// Scalar overlap shift in `F + \lambda S`.
    pub(super) lambda: T,
    /// Worker index and worker count for target rows.
    pub(super) partition: (usize, usize),
}

/// Build same-parent orthogonal one-body action metadata for parent `P`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `data`: Shared NOCI determinant data.
/// - `parent_id`: Parent `P`.
/// # Returns
/// - `OrthogonalOneBodyBlock`: Orthogonal same-parent block with occupation lookup data.
pub(super) fn build_orthogonal_one_body_block<T: NOCIScalar>(
    spin: &SpinFactorisation,
    data: &NOCIData<'_, T>,
    parent_id: usize,
) -> OrthogonalOneBodyBlock {
    let parent = &spin.parents[parent_id];
    let mut opos = HashMap::new();
    let mut groups = Vec::with_capacity(parent.oreps.len());

    for &det in &parent.oreps {
        let state = &data.basis[det];
        opos.insert((state.oa, state.ob), groups.len());
        groups.push(OrthogonalOccupationGroup {
            targets: Vec::new(),
        });
    }

    for entry in &parent.entries {
        let oid = parent.oids[entry.det - parent.first_det];
        groups[oid].targets.push(OrthogonalTarget {
            det: entry.det,
            a: entry.a,
        });
    }

    OrthogonalOneBodyBlock {
        parent: parent_id,
        opos,
        groups,
    }
}

/// Apply same-parent orthogonal `Y^P += (F^{PP}+\lambda S^{PP})D^P`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `block`: Same-parent orthogonal action data.
/// - `x`: Source determinant vector.
/// - `y`: Output determinant vector to accumulate.
/// - `context`: Shared determinant, Fock, shift and worker data.
/// # Returns
/// - `()`: Adds this same-parent contribution into `y`.
pub(super) fn apply_one_body_orthogonal<T: NOCIScalar>(
    spin: &SpinFactorisation,
    block: &OrthogonalOneBodyBlock,
    x: &[T],
    y: &mut [T],
    context: &OrthogonalApplyContext<'_, T>,
) {
    let zero = T::from_real(0.0);
    let source = &spin.parents[block.parent];

    for entry in &source.entries {
        let xe = x[entry.det];
        if xe == zero {
            continue;
        }
        let sdet = &context.data.basis[entry.det];
        let oid = source.oids[entry.det - source.first_det];
        scatter_orthogonal_group(&block.groups[oid], entry.det, xe, y, context);

        apply_orthogonal_alpha_singles(block, entry.det, sdet.oa, sdet.ob, xe, y, context);
        apply_orthogonal_beta_singles(block, entry.det, sdet.oa, sdet.ob, xe, y, context);
    }
}

/// Scatter one same-occupation orthogonal contribution to assigned target rows.
/// # Arguments:
/// - `group`: Target determinants with the same occupation pair.
/// - `source_det`: Source determinant index.
/// - `xe`: Source vector coefficient.
/// - `y`: Output determinant vector to accumulate.
/// - `context`: Shared determinant, Fock, shift and worker data.
/// # Returns
/// - `()`: Adds same-occupation contributions into `y`.
fn scatter_orthogonal_group<T: NOCIScalar>(
    group: &OrthogonalOccupationGroup,
    source_det: usize,
    xe: T,
    y: &mut [T],
    context: &OrthogonalApplyContext<'_, T>,
) {
    let source = &context.data.basis[source_det];
    let (worker, nworker) = context.partition;
    for target in &group.targets {
        if target.a % nworker == worker {
            let target_det = &context.data.basis[target.det];
            let f = calculate_f_pair_orthogonal(context.cache, target_det, source);
            let s = calculate_s_pair_orthogonal(target_det, source);
            y[target.det] += (f + context.lambda * s) * xe;
        }
    }
}

/// Apply all alpha single-excitation orthogonal Fock couplings from one source determinant.
/// # Arguments:
/// - `orthogonal`: Parent-local occupation lookup.
/// - `source_det`: Source determinant index.
/// - `oa`: Source alpha occupation bitstring.
/// - `ob`: Source beta occupation bitstring.
/// - `xe`: Source vector coefficient.
/// - `y`: Output determinant vector to accumulate.
/// - `context`: Shared determinant, Fock and worker data.
/// # Returns
/// - `()`: Adds alpha single-excitation Fock contributions into `y`.
fn apply_orthogonal_alpha_singles<T: NOCIScalar>(
    orthogonal: &OrthogonalOneBodyBlock,
    source_det: usize,
    oa: u128,
    ob: u128,
    xe: T,
    y: &mut [T],
    context: &OrthogonalApplyContext<'_, T>,
) {
    let (worker, nworker) = context.partition;
    let source = &context.data.basis[source_det];
    let nmo = context.cache.fa.nrows();
    let mut holes = oa;
    while holes != 0 {
        let hole = holes.trailing_zeros() as usize;
        holes &= holes - 1;
        for part in 0..nmo {
            if ((oa >> part) & 1) == 1 {
                continue;
            }
            let target_oa = (oa & !(1u128 << hole)) | (1u128 << part);
            let Some(&opos) = orthogonal.opos.get(&(target_oa, ob)) else {
                continue;
            };
            for target in &orthogonal.groups[opos].targets {
                if target.a % nworker == worker {
                    let target_det = &context.data.basis[target.det];
                    y[target.det] +=
                        calculate_f_pair_orthogonal(context.cache, target_det, source) * xe;
                }
            }
        }
    }
}

/// Apply all beta single-excitation orthogonal Fock couplings from one source determinant.
/// # Arguments:
/// - `orthogonal`: Parent-local occupation lookup.
/// - `source_det`: Source determinant index.
/// - `oa`: Source alpha occupation bitstring.
/// - `ob`: Source beta occupation bitstring.
/// - `xe`: Source vector coefficient.
/// - `y`: Output determinant vector to accumulate.
/// - `context`: Shared determinant, Fock and worker data.
/// # Returns
/// - `()`: Adds beta single-excitation Fock contributions into `y`.
fn apply_orthogonal_beta_singles<T: NOCIScalar>(
    orthogonal: &OrthogonalOneBodyBlock,
    source_det: usize,
    oa: u128,
    ob: u128,
    xe: T,
    y: &mut [T],
    context: &OrthogonalApplyContext<'_, T>,
) {
    let (worker, nworker) = context.partition;
    let source = &context.data.basis[source_det];
    let nmo = context.cache.fb.nrows();
    let mut holes = ob;
    while holes != 0 {
        let hole = holes.trailing_zeros() as usize;
        holes &= holes - 1;
        for part in 0..nmo {
            if ((ob >> part) & 1) == 1 {
                continue;
            }
            let target_ob = (ob & !(1u128 << hole)) | (1u128 << part);
            let Some(&opos) = orthogonal.opos.get(&(oa, target_ob)) else {
                continue;
            };
            for target in &orthogonal.groups[opos].targets {
                if target.a % nworker == worker {
                    let target_det = &context.data.basis[target.det];
                    y[target.det] +=
                        calculate_f_pair_orthogonal(context.cache, target_det, source) * xe;
                }
            }
        }
    }
}
