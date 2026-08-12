// noci/factorise/onebody/plan.rs
//! Shared topology plan for factorised one-body NOCI operator contractions.

// Crate-root imports.
use crate::noci::types::{FockData, NOCIScalar};

// Parent/sibling imports.
use super::super::{SpinFactorisation, ordered_parent_pair};

/// Target-component panel width for transient one-body factor construction and contraction.
pub(super) const PANEL_ROWS: usize = 512;

/// Dense one-body contraction order for one parent pair.
#[derive(Clone, Copy)]
pub(crate) enum OneBodyContraction {
    /// Form alpha-first intermediates `T^F_{\bar a b}` and `T^S_{\bar a b}`.
    AFirst,
    /// Form beta-first intermediates `U^F_{a\bar b}` and `U^S_{a\bar b}`.
    BFirst,
}

/// Shared one-body block classification and ordered parent-pair topology.
pub(crate) enum OneBodyBlockPlan {
    /// Same-parent standard Slater-Condon sparse one-body action.
    Orthogonal {
        /// Parent `P`.
        parent: usize,
    },
    /// Spin-factorised nonorthogonal one-body action.
    NonOrthogonal {
        /// Target parent `Q`.
        target_parent: usize,
        /// Source parent `P`.
        source_parent: usize,
        /// Left parent in the ordered Wick pair.
        lp: usize,
        /// Greater parent in the ordered Wick pair.
        gp: usize,
        /// Whether target parent `Q` is the left parent in the ordered Wick pair.
        target_left: bool,
        /// Number of target alpha rows.
        nta: usize,
        /// Number of target beta rows.
        ntb: usize,
        /// Number of source alpha columns.
        nsa: usize,
        /// Number of source beta columns.
        nsb: usize,
        /// Selected dense contraction order for this parent pair.
        contraction: OneBodyContraction,
    },
}

/// Shared one-body block plan for CPU and GPU backends.
pub(crate) struct OneBodyPlan {
    /// Parent-pair block plans indexed as `Q * nparent + P`.
    pub(crate) blocks: Vec<OneBodyBlockPlan>,
    /// Number of parent references.
    pub(crate) nparent: usize,
}

impl OneBodyPlan {
    /// Build shared one-body block topology for CPU and GPU backends.
    /// # Arguments:
    /// - `spin`: Shared determinant-space factorisation.
    /// - `fock`: Current generalised-Fock data used to identify orthogonal same-parent blocks.
    /// # Returns
    /// - `OneBodyPlan`: Parent-pair block classification and contraction-order plan.
    pub(crate) fn new<T: NOCIScalar>(
        spin: &SpinFactorisation,
        fock: &FockData<'_, T>,
    ) -> Self {
        let nparent = spin.parents.len();
        let mut blocks = Vec::with_capacity(nparent * nparent);
        for target_parent in 0..nparent {
            let target = &spin.parents[target_parent];
            for source_parent in 0..nparent {
                if target_parent == source_parent
                    && fock.fock_mocache[target_parent].orthogonal_slater_condon
                {
                    blocks.push(OneBodyBlockPlan::Orthogonal {
                        parent: target_parent,
                    });
                    continue;
                }
                let source = &spin.parents[source_parent];
                let nta = target.areps.len();
                let ntb = target.breps.len();
                let nsa = source.areps.len();
                let nsb = source.breps.len();
                let (lp, gp, target_left) = ordered_parent_pair(spin, target_parent, source_parent);
                let contraction = select_one_body_contraction(
                    nta,
                    ntb,
                    nsa,
                    nsb,
                    target.entries.len(),
                    source.entries.len(),
                );
                blocks.push(OneBodyBlockPlan::NonOrthogonal {
                    target_parent,
                    source_parent,
                    lp,
                    gp,
                    target_left,
                    nta,
                    ntb,
                    nsa,
                    nsb,
                    contraction,
                });
            }
        }
        Self { blocks, nparent }
    }
}

/// Select alpha-first or beta-first contraction from dense structural costs.
/// # Arguments:
/// - `nta`: Number of target alpha components.
/// - `ntb`: Number of target beta components.
/// - `nsa`: Number of source alpha components.
/// - `nsb`: Number of source beta components.
/// - `nt`: Number of actual target determinants.
/// - `ns`: Number of actual source determinants.
/// # Returns
/// - `OneBodyContraction`: Lower estimated-cost contraction.
pub(crate) fn select_one_body_contraction(
    nta: usize,
    ntb: usize,
    nsa: usize,
    nsb: usize,
    nt: usize,
    ns: usize,
) -> OneBodyContraction {
    let ca = 2usize
        .saturating_mul(nta)
        .saturating_mul(ns)
        .saturating_add(2usize.saturating_mul(nt).saturating_mul(nsb));
    let cb = 2usize
        .saturating_mul(ntb)
        .saturating_mul(ns)
        .saturating_add(2usize.saturating_mul(nt).saturating_mul(nsa));
    if ca <= cb {
        OneBodyContraction::AFirst
    } else {
        OneBodyContraction::BFirst
    }
}
