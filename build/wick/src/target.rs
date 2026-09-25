// target.rs
//! Appendix C metric blocks of Lee and Tew, used to check the generated FOIS metric.
//!
//! Each block is written over the free indices of its left and right excitations. Generated
//! and target blocks are compared in canonical form, modulo the spin-\tfrac12 cumulant
//! relations under which different spin-free representations of one block are equal.
//!
//! # References
//!
//! - Lee and Tew, *Spin-free Generalised Normal Ordered Coupled Cluster*, arXiv:2507.13472
//!   (2025), Appendix C.

// External crate imports.
use num_rational::Ratio;
use smallvec::SmallVec;

// Crate-root imports.
use crate::canon::{self, Factor, Form, Key};
use crate::specs::{self, Space};
use crate::{emit, reduce, spin};

/// One named orbital index.
#[derive(Clone, Copy, Debug)]
struct Idx {
    /// Symbolic name.
    name: &'static str,
    /// Orbital space.
    space: Space,
}

/// One Kronecker delta.
#[derive(Clone, Copy, Debug)]
struct Delta(Idx, Idx);

/// One spin-free tensor factor.
#[derive(Clone, Debug)]
struct Tensor {
    /// Spin-free tensor kind id.
    kind: u8,
    /// Upper indices.
    upper: Vec<Idx>,
    /// Lower indices.
    lower: Vec<Idx>,
}

/// One target term.
#[derive(Clone, Debug)]
struct Term {
    /// Rational coefficient.
    coeff: Ratio<i64>,
    /// Delta factors.
    deltas: Vec<Delta>,
    /// Tensor factors.
    tensors: Vec<Tensor>,
}

/// Target expression as a sum of terms.
type Expr = Vec<Term>;

/// Build a core index.
/// # Arguments:
/// - `name`: Symbolic name.
/// # Returns:
/// - `Idx`: Core index.
fn c(name: &'static str) -> Idx {
    Idx {
        name,
        space: Space::Core,
    }
}

/// Build an active index.
/// # Arguments:
/// - `name`: Symbolic name.
/// # Returns:
/// - `Idx`: Active index.
fn a(name: &'static str) -> Idx {
    Idx {
        name,
        space: Space::Active,
    }
}

/// Build a virtual index.
/// # Arguments:
/// - `name`: Symbolic name.
/// # Returns:
/// - `Idx`: Virtual index.
fn v(name: &'static str) -> Idx {
    Idx {
        name,
        space: Space::Virtual,
    }
}

/// Build an integer rational coefficient.
/// # Arguments:
/// - `n`: Numerator.
/// # Returns:
/// - `Ratio<i64>`: Integer coefficient.
fn r(n: i64) -> Ratio<i64> {
    Ratio::from_integer(n)
}

/// Build a rational coefficient.
/// # Arguments:
/// - `n`: Numerator.
/// - `d`: Denominator.
/// # Returns:
/// - `Ratio<i64>`: Rational coefficient.
fn q(
    n: i64,
    d: i64,
) -> Ratio<i64> {
    Ratio::new(n, d)
}

/// Build one term.
/// # Arguments:
/// - `coeff`: Rational coefficient.
/// - `deltas`: Delta factors.
/// - `tensors`: Tensor factors.
/// # Returns:
/// - `Term`: Target term.
fn term(
    coeff: Ratio<i64>,
    deltas: Vec<Delta>,
    tensors: Vec<Tensor>,
) -> Term {
    Term {
        coeff,
        deltas,
        tensors,
    }
}

/// Build a delta.
/// # Arguments:
/// - `left`: Left index.
/// - `right`: Right index.
/// # Returns:
/// - `Delta`: Kronecker delta.
fn d(
    left: Idx,
    right: Idx,
) -> Delta {
    Delta(left, right)
}

/// Build one spin-free tensor.
/// # Arguments:
/// - `kind`: Spin-free tensor kind id.
/// - `upper`: Upper indices.
/// - `lower`: Lower indices.
/// # Returns:
/// - `Tensor`: Tensor factor.
fn tensor(
    kind: u8,
    upper: &[Idx],
    lower: &[Idx],
) -> Tensor {
    Tensor {
        kind,
        upper: upper.to_vec(),
        lower: lower.to_vec(),
    }
}

/// Build a one-particle density `\Gamma^u_l`.
/// # Arguments:
/// - `upper`: Upper active index.
/// - `lower`: Lower active index.
/// # Returns:
/// - `Tensor`: Density factor.
fn g(
    upper: Idx,
    lower: Idx,
) -> Tensor {
    tensor(spin::GAMMA, &[upper], &[lower])
}

/// Build a one-hole density `\Theta^u_l`.
/// # Arguments:
/// - `upper`: Upper active index.
/// - `lower`: Lower active index.
/// # Returns:
/// - `Tensor`: Hole-density factor.
fn th(
    upper: Idx,
    lower: Idx,
) -> Tensor {
    tensor(spin::THETA, &[upper], &[lower])
}

/// Build a two-body cumulant `\Lambda^{u_1u_2}_{l_1l_2}`.
/// # Arguments:
/// - `u1`: First upper active index.
/// - `u2`: Second upper active index.
/// - `l1`: First lower active index.
/// - `l2_`: Second lower active index.
/// # Returns:
/// - `Tensor`: Cumulant factor.
fn l2(
    u1: Idx,
    u2: Idx,
    l1: Idx,
    l2_: Idx,
) -> Tensor {
    tensor(spin::LAMBDA2, &[u1, u2], &[l1, l2_])
}

/// Build a three-body cumulant.
/// # Arguments:
/// - `u`: Upper active indices.
/// - `l`: Lower active indices.
/// # Returns:
/// - `Tensor`: Cumulant factor.
fn l3(
    u: [Idx; 3],
    l: [Idx; 3],
) -> Tensor {
    tensor(spin::LAMBDA3, &u, &l)
}

/// Build a four-body cumulant.
/// # Arguments:
/// - `u`: Upper active indices.
/// - `l`: Lower active indices.
/// # Returns:
/// - `Tensor`: Cumulant factor.
fn l4(
    u: [Idx; 4],
    l: [Idx; 4],
) -> Tensor {
    tensor(spin::LAMBDA4, &u, &l)
}

/// Return the Appendix C1 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for C -> A.
fn c1() -> Expr {
    vec![term(
        r(1),
        vec![d(c("i"), c("j"))],
        vec![th(a("v"), a("u"))],
    )]
}

/// Return the Appendix C2 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> V.
fn c2() -> Expr {
    vec![term(r(1), vec![d(v("b"), v("a"))], vec![g(a("t"), a("u"))])]
}

/// Return the Appendix C3 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> A.
fn c3() -> Expr {
    vec![
        term(q(1, 2), vec![], vec![g(a("u"), a("w")), th(a("x"), a("v"))]),
        term(r(1), vec![], vec![l2(a("u"), a("x"), a("v"), a("w"))]),
    ]
}

/// Return the Appendix C4 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> AV.
fn c4() -> Expr {
    vec![
        term(
            r(1),
            vec![d(c("i"), c("j")), d(v("b"), v("a"))],
            vec![g(a("u"), a("w")), th(a("x"), a("v"))],
        ),
        term(
            r(-1),
            vec![d(c("i"), c("j")), d(v("b"), v("a"))],
            vec![l2(a("u"), a("x"), a("w"), a("v"))],
        ),
    ]
}

/// Return the Appendix C5 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> VA.
fn c5() -> Expr {
    vec![
        term(
            r(1),
            vec![d(c("i"), c("j")), d(v("b"), v("a"))],
            vec![g(a("u"), a("w")), th(a("x"), a("v"))],
        ),
        term(
            r(2),
            vec![d(c("i"), c("j")), d(v("b"), v("a"))],
            vec![l2(a("u"), a("x"), a("v"), a("w"))],
        ),
    ]
}

/// Return the Appendix C6 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> VV.
fn c6() -> Expr {
    vec![
        term(
            r(2),
            vec![d(c("i"), c("j")), d(v("d"), v("b")), d(v("c"), v("a"))],
            vec![g(a("u"), a("v"))],
        ),
        term(
            r(-1),
            vec![d(c("i"), c("j")), d(v("d"), v("a")), d(v("c"), v("b"))],
            vec![g(a("u"), a("v"))],
        ),
    ]
}

/// Return the Appendix C7 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CC -> AV.
fn c7() -> Expr {
    vec![
        term(
            r(2),
            vec![d(v("b"), v("a")), d(c("i"), c("k")), d(c("j"), c("l"))],
            vec![th(a("v"), a("u"))],
        ),
        term(
            r(-1),
            vec![d(v("b"), v("a")), d(c("i"), c("l")), d(c("j"), c("k"))],
            vec![th(a("v"), a("u"))],
        ),
    ]
}

/// Return the Appendix C8 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CC -> AA.
fn c8() -> Expr {
    vec![
        term(
            r(1),
            vec![d(c("i"), c("k")), d(c("j"), c("l"))],
            vec![th(a("w"), a("u")), th(a("x"), a("v"))],
        ),
        term(
            q(-1, 2),
            vec![d(c("i"), c("k")), d(c("j"), c("l"))],
            vec![th(a("w"), a("v")), th(a("x"), a("u"))],
        ),
        term(
            r(1),
            vec![d(c("i"), c("k")), d(c("j"), c("l"))],
            vec![l2(a("w"), a("x"), a("u"), a("v"))],
        ),
        term(
            r(1),
            vec![d(c("i"), c("l")), d(c("j"), c("k"))],
            vec![th(a("w"), a("v")), th(a("x"), a("u"))],
        ),
        term(
            q(-1, 2),
            vec![d(c("i"), c("l")), d(c("j"), c("k"))],
            vec![th(a("w"), a("u")), th(a("x"), a("v"))],
        ),
        term(
            r(1),
            vec![d(c("i"), c("l")), d(c("j"), c("k"))],
            vec![l2(a("w"), a("x"), a("v"), a("u"))],
        ),
    ]
}

/// Return the Appendix C9 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> AA.
fn c9() -> Expr {
    // Appendix C prints C9 without this factor, but the CA -> AA overlap
    // contains one core contraction and Wick evaluation gives delta_i_j.
    let delta = vec![d(c("i"), c("j"))];

    vec![
        term(
            q(1, 2),
            delta.clone(),
            vec![g(a("u"), a("x")), th(a("y"), a("v")), th(a("z"), a("w"))],
        ),
        term(
            q(-1, 4),
            delta.clone(),
            vec![g(a("u"), a("x")), th(a("y"), a("w")), th(a("z"), a("v"))],
        ),
        term(
            q(1, 2),
            delta.clone(),
            vec![g(a("u"), a("x")), l2(a("y"), a("z"), a("v"), a("w"))],
        ),
        term(
            r(1),
            delta.clone(),
            vec![th(a("y"), a("v")), l2(a("u"), a("z"), a("w"), a("x"))],
        ),
        term(
            q(-1, 2),
            delta.clone(),
            vec![th(a("y"), a("w")), l2(a("u"), a("z"), a("v"), a("x"))],
        ),
        term(
            q(-1, 2),
            delta.clone(),
            vec![th(a("z"), a("v")), l2(a("u"), a("y"), a("w"), a("x"))],
        ),
        term(
            q(-1, 2),
            delta.clone(),
            vec![th(a("z"), a("w")), l2(a("u"), a("y"), a("x"), a("v"))],
        ),
        term(
            r(-1),
            delta,
            vec![l3([a("u"), a("y"), a("z")], [a("w"), a("v"), a("x")])],
        ),
    ]
}

/// Return the Appendix C10 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for AA -> AV.
fn c10() -> Expr {
    let delta = vec![d(v("b"), v("a"))];

    vec![
        term(
            q(1, 2),
            delta.clone(),
            vec![th(a("z"), a("v")), g(a("t"), a("x")), g(a("u"), a("y"))],
        ),
        term(
            q(-1, 4),
            delta.clone(),
            vec![th(a("z"), a("v")), g(a("t"), a("y")), g(a("u"), a("x"))],
        ),
        term(
            q(1, 2),
            delta.clone(),
            vec![th(a("z"), a("v")), l2(a("t"), a("u"), a("x"), a("y"))],
        ),
        term(
            q(-1, 2),
            delta.clone(),
            vec![g(a("t"), a("x")), l2(a("u"), a("z"), a("y"), a("v"))],
        ),
        term(
            q(-1, 2),
            delta.clone(),
            vec![g(a("t"), a("y")), l2(a("u"), a("z"), a("v"), a("x"))],
        ),
        term(
            r(1),
            delta.clone(),
            vec![g(a("u"), a("y")), l2(a("t"), a("z"), a("v"), a("x"))],
        ),
        term(
            q(-1, 2),
            delta.clone(),
            vec![g(a("u"), a("x")), l2(a("t"), a("z"), a("v"), a("y"))],
        ),
        term(
            r(1),
            delta,
            vec![l3([a("t"), a("u"), a("z")], [a("v"), a("y"), a("x")])],
        ),
    ]
}

/// Return the Appendix C11 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for AA -> VV.
fn c11() -> Expr {
    let d1 = vec![d(v("d"), v("b")), d(v("c"), v("a"))];
    let d2 = vec![d(v("c"), v("b")), d(v("d"), v("a"))];

    vec![
        term(r(1), d1.clone(), vec![g(a("t"), a("v")), g(a("u"), a("w"))]),
        term(
            q(-1, 2),
            d1.clone(),
            vec![g(a("t"), a("w")), g(a("u"), a("v"))],
        ),
        term(r(1), d1, vec![l2(a("t"), a("u"), a("v"), a("w"))]),
        term(r(1), d2.clone(), vec![g(a("u"), a("v")), g(a("t"), a("w"))]),
        term(
            q(-1, 2),
            d2.clone(),
            vec![g(a("u"), a("w")), g(a("t"), a("v"))],
        ),
        term(r(1), d2, vec![l2(a("u"), a("t"), a("v"), a("w"))]),
    ]
}

/// Return the Appendix C12 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for AA -> AA.
fn c12() -> Expr {
    vec![
        term(
            r(1),
            vec![],
            vec![l4(
                [a("p"), a("r"), a("t"), a("v")],
                [a("q"), a("s"), a("u"), a("w")],
            )],
        ),
        term(
            q(1, 2),
            vec![],
            vec![
                th(a("v"), a("s")),
                l3([a("p"), a("r"), a("t")], [a("q"), a("w"), a("u")]),
            ],
        ),
        term(
            q(1, 2),
            vec![],
            vec![
                th(a("v"), a("q")),
                l3([a("p"), a("r"), a("t")], [a("w"), a("s"), a("u")]),
            ],
        ),
        term(
            q(1, 2),
            vec![],
            vec![
                th(a("t"), a("s")),
                l3([a("p"), a("r"), a("v")], [a("q"), a("u"), a("w")]),
            ],
        ),
        term(
            q(1, 2),
            vec![],
            vec![
                th(a("t"), a("q")),
                l3([a("p"), a("r"), a("v")], [a("u"), a("s"), a("w")]),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                g(a("r"), a("u")),
                l3([a("p"), a("t"), a("v")], [a("q"), a("s"), a("w")]),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                g(a("r"), a("w")),
                l3([a("p"), a("t"), a("v")], [a("q"), a("u"), a("s")]),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                g(a("p"), a("u")),
                l3([a("r"), a("t"), a("v")], [a("s"), a("q"), a("w")]),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                g(a("p"), a("w")),
                l3([a("r"), a("t"), a("v")], [a("s"), a("u"), a("q")]),
            ],
        ),
        term(
            q(1, 4),
            vec![],
            vec![
                th(a("t"), a("q")),
                th(a("v"), a("s")),
                g(a("p"), a("u")),
                g(a("r"), a("w")),
            ],
        ),
        term(
            q(-1, 8),
            vec![],
            vec![
                th(a("t"), a("q")),
                th(a("v"), a("s")),
                g(a("p"), a("w")),
                g(a("r"), a("u")),
            ],
        ),
        term(
            q(1, 4),
            vec![],
            vec![
                th(a("t"), a("q")),
                th(a("v"), a("s")),
                l2(a("p"), a("r"), a("u"), a("w")),
            ],
        ),
        term(
            q(1, 4),
            vec![],
            vec![
                th(a("t"), a("s")),
                th(a("v"), a("q")),
                g(a("p"), a("w")),
                g(a("r"), a("u")),
            ],
        ),
        term(
            q(-1, 8),
            vec![],
            vec![
                th(a("t"), a("s")),
                th(a("v"), a("q")),
                g(a("p"), a("u")),
                g(a("r"), a("w")),
            ],
        ),
        term(
            q(1, 4),
            vec![],
            vec![
                th(a("t"), a("s")),
                th(a("v"), a("q")),
                l2(a("p"), a("r"), a("w"), a("u")),
            ],
        ),
        term(
            q(1, 4),
            vec![],
            vec![
                g(a("p"), a("u")),
                g(a("r"), a("w")),
                l2(a("t"), a("v"), a("q"), a("s")),
            ],
        ),
        term(
            q(1, 4),
            vec![],
            vec![
                g(a("p"), a("w")),
                g(a("r"), a("u")),
                l2(a("t"), a("v"), a("s"), a("q")),
            ],
        ),
        term(
            q(1, 3),
            vec![],
            vec![
                l2(a("t"), a("v"), a("q"), a("s")),
                l2(a("p"), a("r"), a("u"), a("w")),
            ],
        ),
        term(
            q(1, 3),
            vec![],
            vec![
                l2(a("t"), a("v"), a("s"), a("q")),
                l2(a("p"), a("r"), a("w"), a("u")),
            ],
        ),
        term(
            q(1, 6),
            vec![],
            vec![
                l2(a("t"), a("v"), a("q"), a("s")),
                l2(a("p"), a("r"), a("w"), a("u")),
            ],
        ),
        term(
            q(1, 6),
            vec![],
            vec![
                l2(a("t"), a("v"), a("s"), a("q")),
                l2(a("p"), a("r"), a("u"), a("w")),
            ],
        ),
        term(
            q(1, 2),
            vec![],
            vec![
                th(a("v"), a("s")),
                g(a("r"), a("w")),
                l2(a("p"), a("t"), a("q"), a("u")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("v"), a("s")),
                g(a("r"), a("u")),
                l2(a("p"), a("t"), a("q"), a("w")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("v"), a("q")),
                g(a("r"), a("w")),
                l2(a("p"), a("t"), a("s"), a("u")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("v"), a("q")),
                g(a("r"), a("u")),
                l2(a("p"), a("t"), a("w"), a("s")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("t"), a("s")),
                g(a("r"), a("w")),
                l2(a("p"), a("v"), a("q"), a("u")),
            ],
        ),
        term(
            q(1, 2),
            vec![],
            vec![
                th(a("t"), a("s")),
                g(a("r"), a("u")),
                l2(a("p"), a("v"), a("q"), a("w")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("t"), a("q")),
                g(a("r"), a("w")),
                l2(a("p"), a("v"), a("u"), a("s")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("t"), a("q")),
                g(a("r"), a("u")),
                l2(a("p"), a("v"), a("s"), a("w")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("v"), a("s")),
                g(a("p"), a("w")),
                l2(a("r"), a("t"), a("q"), a("u")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("v"), a("s")),
                g(a("p"), a("u")),
                l2(a("r"), a("t"), a("w"), a("q")),
            ],
        ),
        term(
            q(1, 2),
            vec![],
            vec![
                th(a("v"), a("q")),
                g(a("p"), a("w")),
                l2(a("r"), a("t"), a("s"), a("u")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("v"), a("q")),
                g(a("p"), a("u")),
                l2(a("r"), a("t"), a("s"), a("w")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("t"), a("s")),
                g(a("p"), a("w")),
                l2(a("r"), a("v"), a("u"), a("q")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("t"), a("s")),
                g(a("p"), a("u")),
                l2(a("r"), a("v"), a("q"), a("w")),
            ],
        ),
        term(
            q(-1, 4),
            vec![],
            vec![
                th(a("t"), a("q")),
                g(a("p"), a("w")),
                l2(a("r"), a("v"), a("s"), a("u")),
            ],
        ),
        term(
            q(1, 2),
            vec![],
            vec![
                th(a("t"), a("q")),
                g(a("p"), a("u")),
                l2(a("r"), a("v"), a("s"), a("w")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("p"), a("r"), a("w"), a("s")),
                l2(a("t"), a("v"), a("u"), a("q")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("p"), a("r"), a("u"), a("s")),
                l2(a("t"), a("v"), a("q"), a("w")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("p"), a("r"), a("q"), a("w")),
                l2(a("t"), a("v"), a("u"), a("s")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("p"), a("r"), a("q"), a("u")),
                l2(a("t"), a("v"), a("s"), a("w")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("t"), a("r"), a("q"), a("s")),
                l2(a("p"), a("v"), a("u"), a("w")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("t"), a("r"), a("u"), a("w")),
                l2(a("p"), a("v"), a("q"), a("s")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("v"), a("r"), a("q"), a("s")),
                l2(a("t"), a("p"), a("u"), a("w")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("p"), a("t"), a("q"), a("s")),
                l2(a("r"), a("v"), a("u"), a("w")),
            ],
        ),
        term(
            r(1),
            vec![],
            vec![
                l2(a("r"), a("v"), a("s"), a("w")),
                l2(a("p"), a("t"), a("q"), a("u")),
            ],
        ),
        term(
            r(1),
            vec![],
            vec![
                l2(a("r"), a("t"), a("s"), a("u")),
                l2(a("p"), a("v"), a("q"), a("w")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("r"), a("v"), a("s"), a("u")),
                l2(a("p"), a("t"), a("q"), a("w")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("r"), a("v"), a("q"), a("w")),
                l2(a("p"), a("t"), a("s"), a("u")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("p"), a("v"), a("q"), a("u")),
                l2(a("r"), a("t"), a("s"), a("w")),
            ],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![
                l2(a("r"), a("t"), a("q"), a("u")),
                l2(a("p"), a("v"), a("s"), a("w")),
            ],
        ),
        term(
            q(1, 3),
            vec![],
            vec![
                l2(a("r"), a("v"), a("q"), a("u")),
                l2(a("p"), a("t"), a("s"), a("w")),
            ],
        ),
        term(
            q(1, 3),
            vec![],
            vec![
                l2(a("r"), a("v"), a("u"), a("q")),
                l2(a("p"), a("t"), a("w"), a("s")),
            ],
        ),
        term(
            q(1, 6),
            vec![],
            vec![
                l2(a("r"), a("v"), a("u"), a("q")),
                l2(a("p"), a("t"), a("s"), a("w")),
            ],
        ),
        term(
            q(1, 6),
            vec![],
            vec![
                l2(a("r"), a("v"), a("q"), a("u")),
                l2(a("p"), a("t"), a("w"), a("s")),
            ],
        ),
        term(
            q(1, 3),
            vec![],
            vec![
                l2(a("r"), a("t"), a("q"), a("w")),
                l2(a("p"), a("v"), a("s"), a("u")),
            ],
        ),
        term(
            q(1, 3),
            vec![],
            vec![
                l2(a("r"), a("t"), a("w"), a("q")),
                l2(a("p"), a("v"), a("u"), a("s")),
            ],
        ),
        term(
            q(1, 6),
            vec![],
            vec![
                l2(a("r"), a("t"), a("w"), a("q")),
                l2(a("p"), a("v"), a("s"), a("u")),
            ],
        ),
        term(
            q(1, 6),
            vec![],
            vec![
                l2(a("r"), a("t"), a("q"), a("w")),
                l2(a("p"), a("v"), a("u"), a("s")),
            ],
        ),
    ]
}

/// Return the Appendix C13 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> V / AA -> AV.
fn c13() -> Expr {
    vec![term(
        r(1),
        vec![d(v("b"), v("a"))],
        vec![l2(a("u"), a("x"), a("w"), a("v"))],
    )]
}

/// Return the Appendix C14 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for C -> A / CA -> AA.
fn c14() -> Expr {
    vec![term(
        r(-1),
        vec![d(c("i"), c("j"))],
        vec![l2(a("w"), a("x"), a("u"), a("v"))],
    )]
}

/// Return the Appendix C15 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> A / AA -> AA.
fn c15() -> Expr {
    vec![
        term(
            r(1),
            vec![],
            vec![l3([a("t"), a("y"), a("z")], [a("u"), a("w"), a("x")])],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![g(a("t"), a("w")), l2(a("y"), a("z"), a("u"), a("x"))],
        ),
        term(
            q(-1, 2),
            vec![],
            vec![g(a("t"), a("x")), l2(a("y"), a("z"), a("w"), a("u"))],
        ),
        term(
            q(1, 2),
            vec![],
            vec![th(a("y"), a("u")), l2(a("t"), a("z"), a("w"), a("x"))],
        ),
        term(
            q(1, 2),
            vec![],
            vec![th(a("z"), a("u")), l2(a("t"), a("y"), a("x"), a("w"))],
        ),
    ]
}

/// Return the Appendix C16 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> AV / CA -> VA.
fn c16() -> Expr {
    vec![
        term(
            q(-1, 2),
            vec![d(c("i"), c("j")), d(v("b"), v("a"))],
            vec![g(a("u"), a("x")), th(a("y"), a("w"))],
        ),
        term(
            r(-1),
            vec![d(c("i"), c("j")), d(v("b"), v("a"))],
            vec![l2(a("u"), a("y"), a("w"), a("x"))],
        ),
    ]
}

/// Return the Appendix C target expression of one metric block.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `Option<Expr>`: Target expression, or `None` for blocks not listed in Appendix C.
fn target(name: &str) -> Option<Expr> {
    match name {
        "C1" => Some(c1()),
        "C2" => Some(c2()),
        "C3" => Some(c3()),
        "C4" => Some(c4()),
        "C5" => Some(c5()),
        "C6" => Some(c6()),
        "C7" => Some(c7()),
        "C8" => Some(c8()),
        "C9" => Some(c9()),
        "C10" => Some(c10()),
        "C11" => Some(c11()),
        "C12" => Some(c12()),
        "C13" => Some(c13()),
        "C14" => Some(c14()),
        "C15" => Some(c15()),
        "C16" => Some(c16()),
        _ => None,
    }
}

/// Test whether one generated metric block equals its Appendix C target.
/// Target indices are identified with the block free indices by name, left then right.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `bool`: Whether the generated and target blocks differ by spin relations only.
/// # Panics
/// - Panics if `name` has no Appendix C target or uses an index outside the block.
pub fn check(name: &str) -> bool {
    let want = target(name).unwrap_or_else(|| panic!("no Appendix C target for {name}"));
    let b = specs::block(name);
    let names = [b.lf, b.rf].concat();
    let spaces = names
        .iter()
        .map(|&n| specs::space(n) as u8)
        .collect::<Vec<_>>();
    let id = |x: &Idx| {
        names
            .iter()
            .position(|&n| n == x.name)
            .unwrap_or_else(|| panic!("index {} is not free in {name}", x.name)) as u16
    };

    // Difference between generated and target blocks in canonical form.
    let mut diff = emit::metric(name).terms;
    for t in &want {
        let deltas = t.deltas.iter().map(|x| (spin::DELTA, vec![x.0], vec![x.1]));
        let tensors = t
            .tensors
            .iter()
            .map(|x| (x.kind, x.upper.clone(), x.lower.clone()));
        // Target indices carry their own spaces, so a mislabelled index cannot match.
        let mut own = spaces.clone();
        for x in t.deltas.iter().flat_map(|x| [x.0, x.1]) {
            own[id(&x) as usize] = x.space as u8;
        }
        for x in t
            .tensors
            .iter()
            .flat_map(|x| x.upper.iter().chain(&x.lower))
        {
            own[id(x) as usize] = x.space as u8;
        }
        let form = Form {
            spaces: own,
            nfree: names.len(),
            factors: deltas
                .chain(tensors)
                .map(|(kind, upper, lower)| Factor {
                    kind,
                    sym: spin::sym(kind),
                    upper: upper.iter().map(id).collect::<SmallVec<_>>(),
                    lower: lower.iter().map(id).collect::<SmallVec<_>>(),
                })
                .collect(),
        };
        let (key, sign): (Key, i8) = canon::canonical(&form);
        if sign != 0 {
            *diff.entry(key).or_insert_with(|| r(0)) -= t.coeff * r(sign as i64);
        }
    }
    diff.retain(|_, x| *x != r(0));

    reduce::vanishes(&spaces, &diff)
}
