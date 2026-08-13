// noci/factorise/onebody/cpu/factors.rs
//! CPU factor generation for factorised one-body NOCI operator contractions.

// Standard library imports.
use std::ops::Range;

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::noci::types::{NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksPairView};
use crate::nonorthogonalwicks::{prepare_same, xw_overlap_f};

// Parent/sibling imports.
use super::super::super::SpinFactorisation;
use super::super::super::storage::{OneBodyFactorStorage, OneBodyStoragePlan};
use super::super::plan::{OneBodyContraction, PANEL_ROWS};

/// Cached spin-factorised tables and dimensions for one ordered source-target parent pair `QP`.
pub(super) struct FactorisedOneBodyBlock<T: NOCIScalar> {
    /// Target parent `Q`.
    pub(super) target_parent: usize,
    /// Source parent `P`.
    pub(super) source_parent: usize,
    /// Number of target alpha rows.
    pub(super) nta: usize,
    /// Number of target beta rows.
    pub(super) ntb: usize,
    /// Number of source alpha columns.
    pub(super) nsa: usize,
    /// Number of source beta columns.
    pub(super) nsb: usize,
    /// Selected dense contraction order for this parent pair.
    pub(super) contraction: OneBodyContraction,
    /// Raw `S/F` alpha and beta factor backing.
    pub(super) factors: OneBodyFactorStorage<T>,
}

/// Nonpersistent spin-factorised parent pair `QP`.
pub(super) struct TransientOneBodyBlock {
    /// Target parent `Q`.
    pub(super) target_parent: usize,
    /// Source parent `P`.
    pub(super) source_parent: usize,
    /// Left parent in the ordered Wick pair.
    pub(super) lp: usize,
    /// Greater parent in the ordered Wick pair.
    pub(super) gp: usize,
    /// Whether target parent `Q` is the left parent in the ordered Wick pair.
    pub(super) target_left: bool,
    /// Selected dense contraction order for this parent pair.
    pub(super) contraction: OneBodyContraction,
}

/// Build `S^alpha`, `F^alpha`, `S^beta` and `F^beta` tables for one parent pair `QP`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `data`: Shared NOCI data containing Wick intermediates.
/// - `storage_plan`: Storage allocator for factor tables.
/// - `target_parent`: Target parent `Q`.
/// - `source_parent`: Source parent `P`.
/// - `lp`: Left parent in the ordered Wick pair.
/// - `gp`: Greater parent in the ordered Wick pair.
/// - `target_left`: Whether target parent `Q` is left in the ordered Wick pair.
/// - `nta`: Number of target alpha rows.
/// - `ntb`: Number of target beta rows.
/// - `nsa`: Number of source alpha columns.
/// - `nsb`: Number of source beta columns.
/// - `contraction`: Shared dense contraction order for this parent pair.
/// # Returns
/// - `FactorisedOneBodyBlock<T>`: Cached row-major factor tables for this parent pair.
pub(super) fn build_one_body_factor_tables<T: NOCIScalar>(
    spin: &SpinFactorisation,
    data: &NOCIData<'_, T>,
    storage_plan: &mut OneBodyStoragePlan,
    target_parent: usize,
    source_parent: usize,
    lp: usize,
    gp: usize,
    target_left: bool,
    nta: usize,
    ntb: usize,
    nsa: usize,
    nsb: usize,
    contraction: OneBodyContraction,
) -> FactorisedOneBodyBlock<T> {
    let target = &spin.parents[target_parent];
    let source = &spin.parents[source_parent];
    let pair = data
        .wicks
        .expect("factorised one-body requires Wick intermediates")
        .pair(lp, gp);

    let na = nta
        .checked_mul(nsa)
        .expect("alpha one-body factor length overflow");
    let nb = ntb
        .checked_mul(nsb)
        .expect("beta one-body factor length overflow");
    let mut factors = storage_plan.allocate::<T>(target_parent, source_parent, na, nb);

    for row0 in (0..nta).step_by(PANEL_ROWS) {
        let row1 = (row0 + PANEL_ROWS).min(nta);
        let out = factors.alpha_mut();
        build_spin_one_body_factors(
            &pair,
            data,
            (target.areps.as_slice(), source.areps.as_slice()),
            row0..row1,
            target_left,
            true,
            out,
        );
    }
    factors.flush();

    for row0 in (0..ntb).step_by(PANEL_ROWS) {
        let row1 = (row0 + PANEL_ROWS).min(ntb);
        let out = factors.beta_mut();
        build_spin_one_body_factors(
            &pair,
            data,
            (target.breps.as_slice(), source.breps.as_slice()),
            row0..row1,
            target_left,
            false,
            out,
        );
    }
    factors.flush();

    FactorisedOneBodyBlock {
        target_parent,
        source_parent,
        nta,
        ntb,
        nsa,
        nsb,
        contraction,
        factors,
    }
}

/// Build nonpersistent one-body factor metadata for parent pair `QP`.
/// # Arguments:
/// - `target_parent`: Target parent `Q`.
/// - `source_parent`: Source parent `P`.
/// - `lp`: Left parent in the ordered Wick pair.
/// - `gp`: Greater parent in the ordered Wick pair.
/// - `target_left`: Whether target parent `Q` is left in the ordered Wick pair.
/// - `contraction`: Shared dense contraction order for this parent pair.
/// # Returns
/// - `TransientOneBodyBlock`: Parent-pair metadata for regenerated factor tables.
pub(super) fn build_transient_one_body_block(
    target_parent: usize,
    source_parent: usize,
    lp: usize,
    gp: usize,
    target_left: bool,
    contraction: OneBodyContraction,
) -> TransientOneBodyBlock {
    TransientOneBodyBlock {
        target_parent,
        source_parent,
        lp,
        gp,
        target_left,
        contraction,
    }
}

/// Build same-spin `S` and `F` factor rows from one prepared Wick scratch per component pair.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data.
/// - `reps`: Representative determinants for target and source spin components.
/// - `rows`: Target-row range to fill.
/// - `target_left`: Whether target determinants are left determinants in `pair`.
/// - `alpha`: Whether to build alpha or beta factors.
/// - `out`: Mutable row-major overlap and Fock factor tables.
/// # Returns
/// - `()`: Fills `out` factor tables.
pub(super) fn build_spin_one_body_factors<T: NOCIScalar>(
    pair: &WicksPairView<'_, T>,
    data: &NOCIData<'_, T>,
    reps: (&[usize], &[usize]),
    rows: Range<usize>,
    target_left: bool,
    alpha: bool,
    out: (&mut [T], &mut [T]),
) {
    let (target_reps, source_reps) = reps;
    let nsource = source_reps.len();
    let tol = data.tol;
    let row0 = rows.start;
    let row1 = rows.end;
    out.0[row0 * nsource..row1 * nsource]
        .par_chunks_mut(nsource)
        .zip(out.1[row0 * nsource..row1 * nsource].par_chunks_mut(nsource))
        .zip(target_reps[row0..row1].par_iter())
        .for_each_init(WickScratchSpin::new, |scratch, ((srow, frow), &tdet)| {
            for (col, &sdet) in source_reps.iter().enumerate() {
                let (ldet, gdet) = if target_left {
                    (&data.basis[tdet], &data.basis[sdet])
                } else {
                    (&data.basis[sdet], &data.basis[tdet])
                };
                if alpha {
                    let lex = &ldet.excitation.alpha;
                    let gex = &gdet.excitation.alpha;
                    let phase = T::from_real(ldet.pha * gdet.pha);
                    prepare_same(&pair.aa, lex, gex, &mut scratch.aa);
                    let (s, f) = xw_overlap_f(&pair.aa, lex, gex, &mut scratch.aa, tol);
                    srow[col] = phase * s;
                    frow[col] = phase * f;
                } else {
                    let lex = &ldet.excitation.beta;
                    let gex = &gdet.excitation.beta;
                    let phase = T::from_real(ldet.phb * gdet.phb);
                    prepare_same(&pair.bb, lex, gex, &mut scratch.bb);
                    let (s, f) = xw_overlap_f(&pair.bb, lex, gex, &mut scratch.bb, tol);
                    srow[col] = phase * s;
                    frow[col] = phase * f;
                }
            }
        });
}
