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
fn core_index(name: &'static str) -> Idx {
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
fn active_index(name: &'static str) -> Idx {
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
fn virtual_index(name: &'static str) -> Idx {
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
fn integer_coefficient(n: i64) -> Ratio<i64> {
    Ratio::from_integer(n)
}

/// Build a rational coefficient.
/// # Arguments:
/// - `n`: Numerator.
/// - `d`: Denominator.
/// # Returns:
/// - `Ratio<i64>`: Rational coefficient.
fn rational_coefficient(
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
fn target_term(
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
fn kronecker_delta(
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
fn target_tensor(
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
fn particle_density(
    upper: Idx,
    lower: Idx,
) -> Tensor {
    target_tensor(spin::GAMMA, &[upper], &[lower])
}

/// Build a one-hole density `\Theta^u_l`.
/// # Arguments:
/// - `upper`: Upper active index.
/// - `lower`: Lower active index.
/// # Returns:
/// - `Tensor`: Hole-density factor.
fn hole_density(
    upper: Idx,
    lower: Idx,
) -> Tensor {
    target_tensor(spin::THETA, &[upper], &[lower])
}

/// Build a two-body cumulant `\Lambda^{u_1u_2}_{l_1l_2}`.
/// # Arguments:
/// - `u1`: First upper active index.
/// - `u2`: Second upper active index.
/// - `l1`: First lower active index.
/// - `l2_`: Second lower active index.
/// # Returns:
/// - `Tensor`: Cumulant factor.
fn two_body_cumulant(
    u1: Idx,
    u2: Idx,
    l1: Idx,
    l2_: Idx,
) -> Tensor {
    target_tensor(spin::LAMBDA2, &[u1, u2], &[l1, l2_])
}

/// Build a three-body cumulant.
/// # Arguments:
/// - `u`: Upper active indices.
/// - `l`: Lower active indices.
/// # Returns:
/// - `Tensor`: Cumulant factor.
fn three_body_cumulant(
    u: [Idx; 3],
    l: [Idx; 3],
) -> Tensor {
    target_tensor(spin::LAMBDA3, &u, &l)
}

/// Build a four-body cumulant.
/// # Arguments:
/// - `u`: Upper active indices.
/// - `l`: Lower active indices.
/// # Returns:
/// - `Tensor`: Cumulant factor.
fn four_body_cumulant(
    u: [Idx; 4],
    l: [Idx; 4],
) -> Tensor {
    target_tensor(spin::LAMBDA4, &u, &l)
}

/// Return the Appendix C1 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for C -> A.
fn appendix_c1() -> Expr {
    vec![target_term(
        integer_coefficient(1),
        vec![kronecker_delta(core_index("i"), core_index("j"))],
        vec![hole_density(active_index("v"), active_index("u"))],
    )]
}

/// Return the Appendix C2 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> V.
fn appendix_c2() -> Expr {
    vec![target_term(
        integer_coefficient(1),
        vec![kronecker_delta(virtual_index("b"), virtual_index("a"))],
        vec![particle_density(active_index("t"), active_index("u"))],
    )]
}

/// Return the Appendix C3 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> A.
fn appendix_c3() -> Expr {
    vec![
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                particle_density(active_index("u"), active_index("w")),
                hole_density(active_index("x"), active_index("v")),
            ],
        ),
        target_term(
            integer_coefficient(1),
            vec![],
            vec![two_body_cumulant(
                active_index("u"),
                active_index("x"),
                active_index("v"),
                active_index("w"),
            )],
        ),
    ]
}

/// Return the Appendix C4 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> AV.
fn appendix_c4() -> Expr {
    vec![
        target_term(
            integer_coefficient(1),
            vec![
                kronecker_delta(core_index("i"), core_index("j")),
                kronecker_delta(virtual_index("b"), virtual_index("a")),
            ],
            vec![
                particle_density(active_index("u"), active_index("w")),
                hole_density(active_index("x"), active_index("v")),
            ],
        ),
        target_term(
            integer_coefficient(-1),
            vec![
                kronecker_delta(core_index("i"), core_index("j")),
                kronecker_delta(virtual_index("b"), virtual_index("a")),
            ],
            vec![two_body_cumulant(
                active_index("u"),
                active_index("x"),
                active_index("w"),
                active_index("v"),
            )],
        ),
    ]
}

/// Return the Appendix C5 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> VA.
fn appendix_c5() -> Expr {
    vec![
        target_term(
            integer_coefficient(1),
            vec![
                kronecker_delta(core_index("i"), core_index("j")),
                kronecker_delta(virtual_index("b"), virtual_index("a")),
            ],
            vec![
                particle_density(active_index("u"), active_index("w")),
                hole_density(active_index("x"), active_index("v")),
            ],
        ),
        target_term(
            integer_coefficient(2),
            vec![
                kronecker_delta(core_index("i"), core_index("j")),
                kronecker_delta(virtual_index("b"), virtual_index("a")),
            ],
            vec![two_body_cumulant(
                active_index("u"),
                active_index("x"),
                active_index("v"),
                active_index("w"),
            )],
        ),
    ]
}

/// Return the Appendix C6 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> VV.
fn appendix_c6() -> Expr {
    vec![
        target_term(
            integer_coefficient(2),
            vec![
                kronecker_delta(core_index("i"), core_index("j")),
                kronecker_delta(virtual_index("d"), virtual_index("b")),
                kronecker_delta(virtual_index("c"), virtual_index("a")),
            ],
            vec![particle_density(active_index("u"), active_index("v"))],
        ),
        target_term(
            integer_coefficient(-1),
            vec![
                kronecker_delta(core_index("i"), core_index("j")),
                kronecker_delta(virtual_index("d"), virtual_index("a")),
                kronecker_delta(virtual_index("c"), virtual_index("b")),
            ],
            vec![particle_density(active_index("u"), active_index("v"))],
        ),
    ]
}

/// Return the Appendix C7 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CC -> AV.
fn appendix_c7() -> Expr {
    vec![
        target_term(
            integer_coefficient(2),
            vec![
                kronecker_delta(virtual_index("b"), virtual_index("a")),
                kronecker_delta(core_index("i"), core_index("k")),
                kronecker_delta(core_index("j"), core_index("l")),
            ],
            vec![hole_density(active_index("v"), active_index("u"))],
        ),
        target_term(
            integer_coefficient(-1),
            vec![
                kronecker_delta(virtual_index("b"), virtual_index("a")),
                kronecker_delta(core_index("i"), core_index("l")),
                kronecker_delta(core_index("j"), core_index("k")),
            ],
            vec![hole_density(active_index("v"), active_index("u"))],
        ),
    ]
}

/// Return the Appendix C8 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CC -> AA.
fn appendix_c8() -> Expr {
    vec![
        target_term(
            integer_coefficient(1),
            vec![
                kronecker_delta(core_index("i"), core_index("k")),
                kronecker_delta(core_index("j"), core_index("l")),
            ],
            vec![
                hole_density(active_index("w"), active_index("u")),
                hole_density(active_index("x"), active_index("v")),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![
                kronecker_delta(core_index("i"), core_index("k")),
                kronecker_delta(core_index("j"), core_index("l")),
            ],
            vec![
                hole_density(active_index("w"), active_index("v")),
                hole_density(active_index("x"), active_index("u")),
            ],
        ),
        target_term(
            integer_coefficient(1),
            vec![
                kronecker_delta(core_index("i"), core_index("k")),
                kronecker_delta(core_index("j"), core_index("l")),
            ],
            vec![two_body_cumulant(
                active_index("w"),
                active_index("x"),
                active_index("u"),
                active_index("v"),
            )],
        ),
        target_term(
            integer_coefficient(1),
            vec![
                kronecker_delta(core_index("i"), core_index("l")),
                kronecker_delta(core_index("j"), core_index("k")),
            ],
            vec![
                hole_density(active_index("w"), active_index("v")),
                hole_density(active_index("x"), active_index("u")),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![
                kronecker_delta(core_index("i"), core_index("l")),
                kronecker_delta(core_index("j"), core_index("k")),
            ],
            vec![
                hole_density(active_index("w"), active_index("u")),
                hole_density(active_index("x"), active_index("v")),
            ],
        ),
        target_term(
            integer_coefficient(1),
            vec![
                kronecker_delta(core_index("i"), core_index("l")),
                kronecker_delta(core_index("j"), core_index("k")),
            ],
            vec![two_body_cumulant(
                active_index("w"),
                active_index("x"),
                active_index("v"),
                active_index("u"),
            )],
        ),
    ]
}

/// Return the Appendix C9 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> AA.
fn appendix_c9() -> Expr {
    // Appendix C prints C9 without this factor, but the CA -> AA overlap
    // contains one core contraction and Wick evaluation gives delta_i_j.
    let delta = vec![kronecker_delta(core_index("i"), core_index("j"))];

    vec![
        target_term(
            rational_coefficient(1, 2),
            delta.clone(),
            vec![
                particle_density(active_index("u"), active_index("x")),
                hole_density(active_index("y"), active_index("v")),
                hole_density(active_index("z"), active_index("w")),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            delta.clone(),
            vec![
                particle_density(active_index("u"), active_index("x")),
                hole_density(active_index("y"), active_index("w")),
                hole_density(active_index("z"), active_index("v")),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            delta.clone(),
            vec![
                particle_density(active_index("u"), active_index("x")),
                two_body_cumulant(
                    active_index("y"),
                    active_index("z"),
                    active_index("v"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            integer_coefficient(1),
            delta.clone(),
            vec![
                hole_density(active_index("y"), active_index("v")),
                two_body_cumulant(
                    active_index("u"),
                    active_index("z"),
                    active_index("w"),
                    active_index("x"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            delta.clone(),
            vec![
                hole_density(active_index("y"), active_index("w")),
                two_body_cumulant(
                    active_index("u"),
                    active_index("z"),
                    active_index("v"),
                    active_index("x"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            delta.clone(),
            vec![
                hole_density(active_index("z"), active_index("v")),
                two_body_cumulant(
                    active_index("u"),
                    active_index("y"),
                    active_index("w"),
                    active_index("x"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            delta.clone(),
            vec![
                hole_density(active_index("z"), active_index("w")),
                two_body_cumulant(
                    active_index("u"),
                    active_index("y"),
                    active_index("x"),
                    active_index("v"),
                ),
            ],
        ),
        target_term(
            integer_coefficient(-1),
            delta,
            vec![three_body_cumulant(
                [active_index("u"), active_index("y"), active_index("z")],
                [active_index("w"), active_index("v"), active_index("x")],
            )],
        ),
    ]
}

/// Return the Appendix C10 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for AA -> AV.
fn appendix_c10() -> Expr {
    let delta = vec![kronecker_delta(virtual_index("b"), virtual_index("a"))];

    vec![
        target_term(
            rational_coefficient(1, 2),
            delta.clone(),
            vec![
                hole_density(active_index("z"), active_index("v")),
                particle_density(active_index("t"), active_index("x")),
                particle_density(active_index("u"), active_index("y")),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            delta.clone(),
            vec![
                hole_density(active_index("z"), active_index("v")),
                particle_density(active_index("t"), active_index("y")),
                particle_density(active_index("u"), active_index("x")),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            delta.clone(),
            vec![
                hole_density(active_index("z"), active_index("v")),
                two_body_cumulant(
                    active_index("t"),
                    active_index("u"),
                    active_index("x"),
                    active_index("y"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            delta.clone(),
            vec![
                particle_density(active_index("t"), active_index("x")),
                two_body_cumulant(
                    active_index("u"),
                    active_index("z"),
                    active_index("y"),
                    active_index("v"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            delta.clone(),
            vec![
                particle_density(active_index("t"), active_index("y")),
                two_body_cumulant(
                    active_index("u"),
                    active_index("z"),
                    active_index("v"),
                    active_index("x"),
                ),
            ],
        ),
        target_term(
            integer_coefficient(1),
            delta.clone(),
            vec![
                particle_density(active_index("u"), active_index("y")),
                two_body_cumulant(
                    active_index("t"),
                    active_index("z"),
                    active_index("v"),
                    active_index("x"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            delta.clone(),
            vec![
                particle_density(active_index("u"), active_index("x")),
                two_body_cumulant(
                    active_index("t"),
                    active_index("z"),
                    active_index("v"),
                    active_index("y"),
                ),
            ],
        ),
        target_term(
            integer_coefficient(1),
            delta,
            vec![three_body_cumulant(
                [active_index("t"), active_index("u"), active_index("z")],
                [active_index("v"), active_index("y"), active_index("x")],
            )],
        ),
    ]
}

/// Return the Appendix C11 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for AA -> VV.
fn appendix_c11() -> Expr {
    let d1 = vec![
        kronecker_delta(virtual_index("d"), virtual_index("b")),
        kronecker_delta(virtual_index("c"), virtual_index("a")),
    ];
    let d2 = vec![
        kronecker_delta(virtual_index("c"), virtual_index("b")),
        kronecker_delta(virtual_index("d"), virtual_index("a")),
    ];

    vec![
        target_term(
            integer_coefficient(1),
            d1.clone(),
            vec![
                particle_density(active_index("t"), active_index("v")),
                particle_density(active_index("u"), active_index("w")),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            d1.clone(),
            vec![
                particle_density(active_index("t"), active_index("w")),
                particle_density(active_index("u"), active_index("v")),
            ],
        ),
        target_term(
            integer_coefficient(1),
            d1,
            vec![two_body_cumulant(
                active_index("t"),
                active_index("u"),
                active_index("v"),
                active_index("w"),
            )],
        ),
        target_term(
            integer_coefficient(1),
            d2.clone(),
            vec![
                particle_density(active_index("u"), active_index("v")),
                particle_density(active_index("t"), active_index("w")),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            d2.clone(),
            vec![
                particle_density(active_index("u"), active_index("w")),
                particle_density(active_index("t"), active_index("v")),
            ],
        ),
        target_term(
            integer_coefficient(1),
            d2,
            vec![two_body_cumulant(
                active_index("u"),
                active_index("t"),
                active_index("v"),
                active_index("w"),
            )],
        ),
    ]
}

/// Return the Appendix C12 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for AA -> AA.
fn appendix_c12() -> Expr {
    vec![
        target_term(
            integer_coefficient(1),
            vec![],
            vec![four_body_cumulant(
                [
                    active_index("p"),
                    active_index("r"),
                    active_index("t"),
                    active_index("v"),
                ],
                [
                    active_index("q"),
                    active_index("s"),
                    active_index("u"),
                    active_index("w"),
                ],
            )],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("s")),
                three_body_cumulant(
                    [active_index("p"), active_index("r"), active_index("t")],
                    [active_index("q"), active_index("w"), active_index("u")],
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("q")),
                three_body_cumulant(
                    [active_index("p"), active_index("r"), active_index("t")],
                    [active_index("w"), active_index("s"), active_index("u")],
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("s")),
                three_body_cumulant(
                    [active_index("p"), active_index("r"), active_index("v")],
                    [active_index("q"), active_index("u"), active_index("w")],
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("q")),
                three_body_cumulant(
                    [active_index("p"), active_index("r"), active_index("v")],
                    [active_index("u"), active_index("s"), active_index("w")],
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                particle_density(active_index("r"), active_index("u")),
                three_body_cumulant(
                    [active_index("p"), active_index("t"), active_index("v")],
                    [active_index("q"), active_index("s"), active_index("w")],
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                particle_density(active_index("r"), active_index("w")),
                three_body_cumulant(
                    [active_index("p"), active_index("t"), active_index("v")],
                    [active_index("q"), active_index("u"), active_index("s")],
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                particle_density(active_index("p"), active_index("u")),
                three_body_cumulant(
                    [active_index("r"), active_index("t"), active_index("v")],
                    [active_index("s"), active_index("q"), active_index("w")],
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                particle_density(active_index("p"), active_index("w")),
                three_body_cumulant(
                    [active_index("r"), active_index("t"), active_index("v")],
                    [active_index("s"), active_index("u"), active_index("q")],
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("q")),
                hole_density(active_index("v"), active_index("s")),
                particle_density(active_index("p"), active_index("u")),
                particle_density(active_index("r"), active_index("w")),
            ],
        ),
        target_term(
            rational_coefficient(-1, 8),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("q")),
                hole_density(active_index("v"), active_index("s")),
                particle_density(active_index("p"), active_index("w")),
                particle_density(active_index("r"), active_index("u")),
            ],
        ),
        target_term(
            rational_coefficient(1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("q")),
                hole_density(active_index("v"), active_index("s")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("u"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("s")),
                hole_density(active_index("v"), active_index("q")),
                particle_density(active_index("p"), active_index("w")),
                particle_density(active_index("r"), active_index("u")),
            ],
        ),
        target_term(
            rational_coefficient(-1, 8),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("s")),
                hole_density(active_index("v"), active_index("q")),
                particle_density(active_index("p"), active_index("u")),
                particle_density(active_index("r"), active_index("w")),
            ],
        ),
        target_term(
            rational_coefficient(1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("s")),
                hole_density(active_index("v"), active_index("q")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("w"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 4),
            vec![],
            vec![
                particle_density(active_index("p"), active_index("u")),
                particle_density(active_index("r"), active_index("w")),
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("q"),
                    active_index("s"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 4),
            vec![],
            vec![
                particle_density(active_index("p"), active_index("w")),
                particle_density(active_index("r"), active_index("u")),
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("s"),
                    active_index("q"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 3),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("q"),
                    active_index("s"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("u"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 3),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("s"),
                    active_index("q"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("w"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 6),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("q"),
                    active_index("s"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("w"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 6),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("s"),
                    active_index("q"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("u"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("s")),
                particle_density(active_index("r"), active_index("w")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("q"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("s")),
                particle_density(active_index("r"), active_index("u")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("q"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("q")),
                particle_density(active_index("r"), active_index("w")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("s"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("q")),
                particle_density(active_index("r"), active_index("u")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("w"),
                    active_index("s"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("s")),
                particle_density(active_index("r"), active_index("w")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("q"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("s")),
                particle_density(active_index("r"), active_index("u")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("q"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("q")),
                particle_density(active_index("r"), active_index("w")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("u"),
                    active_index("s"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("q")),
                particle_density(active_index("r"), active_index("u")),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("s"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("s")),
                particle_density(active_index("p"), active_index("w")),
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("q"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("s")),
                particle_density(active_index("p"), active_index("u")),
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("w"),
                    active_index("q"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("q")),
                particle_density(active_index("p"), active_index("w")),
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("s"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("v"), active_index("q")),
                particle_density(active_index("p"), active_index("u")),
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("s"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("s")),
                particle_density(active_index("p"), active_index("w")),
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("u"),
                    active_index("q"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("s")),
                particle_density(active_index("p"), active_index("u")),
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("q"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 4),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("q")),
                particle_density(active_index("p"), active_index("w")),
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("s"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("t"), active_index("q")),
                particle_density(active_index("p"), active_index("u")),
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("s"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("w"),
                    active_index("s"),
                ),
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("u"),
                    active_index("q"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("u"),
                    active_index("s"),
                ),
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("q"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("q"),
                    active_index("w"),
                ),
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("u"),
                    active_index("s"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("p"),
                    active_index("r"),
                    active_index("q"),
                    active_index("u"),
                ),
                two_body_cumulant(
                    active_index("t"),
                    active_index("v"),
                    active_index("s"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("t"),
                    active_index("r"),
                    active_index("q"),
                    active_index("s"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("u"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("t"),
                    active_index("r"),
                    active_index("u"),
                    active_index("w"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("q"),
                    active_index("s"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("v"),
                    active_index("r"),
                    active_index("q"),
                    active_index("s"),
                ),
                two_body_cumulant(
                    active_index("t"),
                    active_index("p"),
                    active_index("u"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("q"),
                    active_index("s"),
                ),
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("u"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            integer_coefficient(1),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("s"),
                    active_index("w"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("q"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            integer_coefficient(1),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("s"),
                    active_index("u"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("q"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("s"),
                    active_index("u"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("q"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("q"),
                    active_index("w"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("s"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("q"),
                    active_index("u"),
                ),
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("s"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("q"),
                    active_index("u"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("s"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 3),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("q"),
                    active_index("u"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("s"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 3),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("u"),
                    active_index("q"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("w"),
                    active_index("s"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 6),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("u"),
                    active_index("q"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("s"),
                    active_index("w"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 6),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("v"),
                    active_index("q"),
                    active_index("u"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("t"),
                    active_index("w"),
                    active_index("s"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 3),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("q"),
                    active_index("w"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("s"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 3),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("w"),
                    active_index("q"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("u"),
                    active_index("s"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 6),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("w"),
                    active_index("q"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("s"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 6),
            vec![],
            vec![
                two_body_cumulant(
                    active_index("r"),
                    active_index("t"),
                    active_index("q"),
                    active_index("w"),
                ),
                two_body_cumulant(
                    active_index("p"),
                    active_index("v"),
                    active_index("u"),
                    active_index("s"),
                ),
            ],
        ),
    ]
}

/// Return the Appendix C13 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> V / AA -> AV.
fn appendix_c13() -> Expr {
    vec![target_term(
        integer_coefficient(1),
        vec![kronecker_delta(virtual_index("b"), virtual_index("a"))],
        vec![two_body_cumulant(
            active_index("u"),
            active_index("x"),
            active_index("w"),
            active_index("v"),
        )],
    )]
}

/// Return the Appendix C14 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for C -> A / CA -> AA.
fn appendix_c14() -> Expr {
    vec![target_term(
        integer_coefficient(-1),
        vec![kronecker_delta(core_index("i"), core_index("j"))],
        vec![two_body_cumulant(
            active_index("w"),
            active_index("x"),
            active_index("u"),
            active_index("v"),
        )],
    )]
}

/// Return the Appendix C15 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> A / AA -> AA.
fn appendix_c15() -> Expr {
    vec![
        target_term(
            integer_coefficient(1),
            vec![],
            vec![three_body_cumulant(
                [active_index("t"), active_index("y"), active_index("z")],
                [active_index("u"), active_index("w"), active_index("x")],
            )],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                particle_density(active_index("t"), active_index("w")),
                two_body_cumulant(
                    active_index("y"),
                    active_index("z"),
                    active_index("u"),
                    active_index("x"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(-1, 2),
            vec![],
            vec![
                particle_density(active_index("t"), active_index("x")),
                two_body_cumulant(
                    active_index("y"),
                    active_index("z"),
                    active_index("w"),
                    active_index("u"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("y"), active_index("u")),
                two_body_cumulant(
                    active_index("t"),
                    active_index("z"),
                    active_index("w"),
                    active_index("x"),
                ),
            ],
        ),
        target_term(
            rational_coefficient(1, 2),
            vec![],
            vec![
                hole_density(active_index("z"), active_index("u")),
                two_body_cumulant(
                    active_index("t"),
                    active_index("y"),
                    active_index("x"),
                    active_index("w"),
                ),
            ],
        ),
    ]
}

/// Return the Appendix C16 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> AV / CA -> VA.
fn appendix_c16() -> Expr {
    vec![
        target_term(
            rational_coefficient(-1, 2),
            vec![
                kronecker_delta(core_index("i"), core_index("j")),
                kronecker_delta(virtual_index("b"), virtual_index("a")),
            ],
            vec![
                particle_density(active_index("u"), active_index("x")),
                hole_density(active_index("y"), active_index("w")),
            ],
        ),
        target_term(
            integer_coefficient(-1),
            vec![
                kronecker_delta(core_index("i"), core_index("j")),
                kronecker_delta(virtual_index("b"), virtual_index("a")),
            ],
            vec![two_body_cumulant(
                active_index("u"),
                active_index("y"),
                active_index("w"),
                active_index("x"),
            )],
        ),
    ]
}

/// Return the Appendix C target expression of one metric block.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `Option<Expr>`: Target expression, or `None` for blocks not listed in Appendix C.
fn appendix_c_block(name: &str) -> Option<Expr> {
    match name {
        "C1" => Some(appendix_c1()),
        "C2" => Some(appendix_c2()),
        "C3" => Some(appendix_c3()),
        "C4" => Some(appendix_c4()),
        "C5" => Some(appendix_c5()),
        "C6" => Some(appendix_c6()),
        "C7" => Some(appendix_c7()),
        "C8" => Some(appendix_c8()),
        "C9" => Some(appendix_c9()),
        "C10" => Some(appendix_c10()),
        "C11" => Some(appendix_c11()),
        "C12" => Some(appendix_c12()),
        "C13" => Some(appendix_c13()),
        "C14" => Some(appendix_c14()),
        "C15" => Some(appendix_c15()),
        "C16" => Some(appendix_c16()),
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
pub fn matches_appendix_c(name: &str) -> bool {
    let want = appendix_c_block(name).unwrap_or_else(|| panic!("no Appendix C target for {name}"));
    let b = specs::metric_block_spec(name);
    let names = [b.lf, b.rf].concat();
    let spaces = names
        .iter()
        .map(|&n| specs::index_space(n) as u8)
        .collect::<Vec<_>>();
    let id = |x: &Idx| {
        names
            .iter()
            .position(|&n| n == x.name)
            .unwrap_or_else(|| panic!("index {} is not free in {name}", x.name)) as u16
    };

    // Difference between generated and target blocks in canonical form.
    let mut diff = emit::metric_table(name).terms;
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
                    sym: spin::slot_symmetry(kind),
                    upper: upper.iter().map(id).collect::<SmallVec<_>>(),
                    lower: lower.iter().map(id).collect::<SmallVec<_>>(),
                })
                .collect(),
        };
        let (key, sign): (Key, i8) = canon::canonical_key(&form);
        if sign != 0 {
            *diff.entry(key).or_insert_with(|| integer_coefficient(0)) -=
                t.coeff * integer_coefficient(sign as i64);
        }
    }
    diff.retain(|_, x| *x != integer_coefficient(0));

    reduce::vanishes_modulo_relations(&spaces, &diff)
}
