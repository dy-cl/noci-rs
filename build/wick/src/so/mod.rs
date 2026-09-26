// so/mod.rs
//! Spin-orbital generalised-normal-ordered Wick contraction.
//!
//! This stage derives the GNOCC equations in an antisymmetrised spin-orbital basis, using
//! exactly the operator and tensor conventions of Wick&D so that its output can be compared
//! term by term. Operators are sums of components
//!
//! `c X^{u_1..u_n}_{l_1..l_m} a^\dagger_{l_1}\cdots a^\dagger_{l_m} a_{u_n}\cdots a_{u_1},`
//!
//! the reference defines core (occupied), active (general) and virtual (unoccupied) spaces, and
//! contractions produce Kronecker deltas, one-body densities `\gamma^p_q` and `\eta^p_q`, and
//! cumulants `\lambda_k` for `k \le 4`. The residual of one excitation class is
//!
//! `R = \langle\Phi|\hat\tau^\dagger\hat H\{1 + \hat T + \tfrac12\hat T^2\}|\Phi\rangle_c,`
//!
//! where the cluster operators of `\{\hat T^2\}` share one normal-ordered string and every
//! kept diagram is connected.
//!
//! # References
//!
//! - Lee and Tew, *Spin-free Generalised Normal Ordered Coupled Cluster*, arXiv:2507.13472
//!   (2025), Eqs. (34)-(36).
//! - Wick&D: Evangelista, *J. Chem. Phys.* **157**, 064111 (2022).

// Restricted submodules.
pub(crate) mod ops;
pub(crate) mod wick;

// Private submodules.
mod wickd;

// Restricted type re-exports.
pub(crate) use crate::specs::Space;

// Restricted function re-exports.
pub(crate) use wickd::parse_wickd_expression;

// External crate imports.
use num_rational::Ratio;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

// Crate-root imports.
use crate::canon::{self, Factor, Form, Key, Sym};

/// Spin-orbital tensor kinds, numbered for canonical keys.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub(crate) enum Kind {
    /// Residual projector `\tau^\dagger`.
    Bra,
    /// Fock operator `f`.
    Fock,
    /// Antisymmetrised two-electron integral `v`.
    Eri,
    /// Singles amplitude.
    T1,
    /// Antisymmetrised doubles amplitude.
    T2,
    /// One-particle density `\gamma^p_q = \langle a^\dagger_p a_q\rangle`.
    Gamma,
    /// One-hole density `\eta^p_q = \langle a_q a^\dagger_p\rangle`.
    Eta,
    /// Two-body cumulant.
    Lambda2,
    /// Three-body cumulant.
    Lambda3,
    /// Four-body cumulant.
    Lambda4,
    /// Metric excitation placeholder `\tau`.
    Ket,
}

/// One spin-orbital tensor factor over integer index ids.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Tensor {
    /// Tensor kind.
    pub(crate) kind: Kind,
    /// Upper index ids: annihilated indices of operator tensors, creators of contractions.
    pub(crate) upper: SmallVec<[u16; 4]>,
    /// Lower index ids: created indices of operator tensors, annihilators of contractions.
    pub(crate) lower: SmallVec<[u16; 4]>,
}

/// One spin-orbital term: a coefficient times a product of tensors, summed over every index.
#[derive(Clone, Debug)]
pub(crate) struct Term {
    /// Exact coefficient.
    pub(crate) coeff: Ratio<i64>,
    /// Orbital space of every index id.
    pub(crate) spaces: Vec<Space>,
    /// Tensor factors.
    pub(crate) tensors: Vec<Tensor>,
}

/// Canonically combined spin-orbital expression.
pub(crate) type Expr = FxHashMap<Key, Ratio<i64>>;

/// Return the canonical key and sign of one spin-orbital term.
/// Every spin-orbital tensor is antisymmetric in its upper and in its lower indices, and every
/// index is summed, so the term has no free indices.
/// # Arguments:
/// - `t`: Spin-orbital term.
/// # Returns:
/// - `(Key, i8)`: Canonical key and sign, `0` when the term vanishes by symmetry.
pub(crate) fn canonical_term_key(t: &Term) -> (Key, i8) {
    let form = Form {
        spaces: t.spaces.iter().map(|&s| s as u8).collect(),
        nfree: 0,
        factors: t
            .tensors
            .iter()
            .map(|x| Factor {
                kind: x.kind as u8,
                sym: Sym::Antisymmetric,
                upper: x.upper.clone(),
                lower: x.lower.clone(),
            })
            .collect(),
    };

    canon::canonical_key(&form)
}

/// Add one term to a canonical expression, dropping cancelled keys.
/// # Arguments:
/// - `acc`: Canonical expression.
/// - `t`: Spin-orbital term.
/// # Returns:
/// - `()`: Mutates `acc`.
pub(crate) fn add_term(
    acc: &mut Expr,
    t: &Term,
) {
    let (k, sign) = canonical_term_key(t);

    if sign == 0 || t.coeff == Ratio::from_integer(0) {
        return;
    }

    let c = acc.entry(k).or_insert_with(|| Ratio::from_integer(0));
    *c += t.coeff * Ratio::from_integer(sign as i64);
}

/// Combine two canonical expressions, dropping cancelled keys.
/// # Arguments:
/// - `acc`: Destination expression.
/// - `other`: Expression to add.
/// # Returns:
/// - `()`: Mutates `acc`.
pub(crate) fn merge_expressions(
    acc: &mut Expr,
    other: Expr,
) {
    for (k, c) in other {
        *acc.entry(k).or_insert_with(|| Ratio::from_integer(0)) += c;
    }
    acc.retain(|_, c| *c != Ratio::from_integer(0));
}

/// Agreement between generated and reference spin-orbital residuals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Comparison {
    /// Number of generated canonical terms.
    pub generated: usize,
    /// Number of reference canonical terms.
    pub reference: usize,
    /// Terms present in both with equal coefficients.
    pub matching: usize,
    /// Terms present in both with different coefficients.
    pub mismatched: usize,
    /// Reference terms absent from the generated residual.
    pub missing: usize,
    /// Generated terms absent from the reference residual.
    pub extra: usize,
}

/// Compare the generated spin-orbital residual of one class with a Wick&D reference.
/// # Arguments:
/// - `order`: Order in `T`.
/// - `class`: Spin-orbital excitation class name.
/// - `reference`: Wick&D expression text for the same residual.
/// # Returns:
/// - `Comparison`: Term-by-term agreement of the two canonical expressions.
/// # Panics
/// - Panics if `class` is not a known excitation class.
pub fn compare_with_wickd(
    order: usize,
    class: &str,
    reference: &str,
) -> Comparison {
    let bra = ops::projector_for_class(class)
        .unwrap_or_else(|| panic!("unknown excitation class {class}"));
    let ours = wick::residual_expression(&bra, order);
    let theirs = parse_wickd_expression(reference);

    let mut out = Comparison {
        generated: ours.len(),
        reference: theirs.len(),
        matching: 0,
        mismatched: 0,
        missing: 0,
        extra: 0,
    };

    for (k, c) in &theirs {
        match ours.get(k) {
            Some(d) if d == c => out.matching += 1,
            Some(_) => out.mismatched += 1,
            None => out.missing += 1,
        }
    }
    out.extra = ours.keys().filter(|k| !theirs.contains_key(*k)).count();

    out
}
