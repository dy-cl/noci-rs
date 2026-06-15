// target.rs

use crate::ir::{a, c, v, Delta, Expr, Rational, Tensor, TensorKind, Term};

/// Build an integer rational coefficient.
/// # Arguments:
/// - `n`: Numerator.
/// # Returns:
/// - `Rational`: Integer coefficient.
fn r(n: i64) -> Rational {
    Rational { num: n, den: 1 }
}

/// Build a rational coefficient.
/// # Arguments:
/// - `n`: Numerator.
/// - `d`: Denominator.
/// # Returns:
/// - `Rational`: Rational coefficient.
fn q(n: i64, d: i64) -> Rational {
    Rational { num: n, den: d }
}

/// Build one term.
/// # Arguments:
/// - `coeff`: Rational coefficient.
/// - `deltas`: Delta factors.
/// - `tensors`: Tensor factors.
/// # Returns:
/// - `Term`: Symbolic term.
fn term(coeff: Rational, deltas: Vec<Delta>, tensors: Vec<Tensor>) -> Term {
    Term { coeff, deltas, tensors }
}

/// Build a delta.
/// # Arguments:
/// - `left`: Left index.
/// - `right`: Right index.
/// # Returns:
/// - `Delta`: Kronecker delta.
fn d(left: crate::ir::Idx, right: crate::ir::Idx) -> Delta {
    Delta { left, right }
}

/// Build a Gamma1 tensor.
/// # Arguments:
/// - `upper`: Upper active index.
/// - `lower`: Lower active index.
/// # Returns:
/// - `Tensor`: Gamma1 tensor.
fn g(upper: crate::ir::Idx, lower: crate::ir::Idx) -> Tensor {
    Tensor { kind: TensorKind::Gamma1, upper: vec![upper], lower: vec![lower] }
}

/// Build a Theta tensor.
/// # Arguments:
/// - `upper`: Upper active index.
/// - `lower`: Lower active index.
/// # Returns:
/// - `Tensor`: Theta tensor.
fn th(upper: crate::ir::Idx, lower: crate::ir::Idx) -> Tensor {
    Tensor { kind: TensorKind::Theta, upper: vec![upper], lower: vec![lower] }
}

/// Build a Lambda2 tensor.
/// # Arguments:
/// - `u1`: First upper active index.
/// - `u2`: Second upper active index.
/// - `l1`: First lower active index.
/// - `l2`: Second lower active index.
/// # Returns:
/// - `Tensor`: Lambda2 tensor.
fn l2(u1: crate::ir::Idx, u2: crate::ir::Idx, l1: crate::ir::Idx, l2_: crate::ir::Idx) -> Tensor {
    Tensor { kind: TensorKind::Lambda2, upper: vec![u1, u2], lower: vec![l1, l2_] }
}

/// Build a Lambda3 tensor.
/// # Arguments:
/// - `u`: Upper active indices.
/// - `l`: Lower active indices.
/// # Returns:
/// - `Tensor`: Lambda3 tensor.
fn l3(u: [crate::ir::Idx; 3], l: [crate::ir::Idx; 3]) -> Tensor {
    Tensor { kind: TensorKind::Lambda3, upper: u.to_vec(), lower: l.to_vec() }
}

/// Build a Lambda4 tensor.
/// # Arguments:
/// - `u`: Upper active indices.
/// - `l`: Lower active indices.
/// # Returns:
/// - `Tensor`: Lambda4 tensor.
fn l4(u: [crate::ir::Idx; 4], l: [crate::ir::Idx; 4]) -> Tensor {
    Tensor { kind: TensorKind::Lambda4, upper: u.to_vec(), lower: l.to_vec() }
}

/// Return the Appendix C1 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for C -> A.
fn c1() -> Expr {
    vec![
        term(r(1), vec![d(c("i"), c("j"))], vec![th(a("v"), a("u"))]),
    ]
}

/// Return the Appendix C2 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> V.
fn c2() -> Expr {
    vec![
        term(r(1), vec![d(v("b"), v("a"))], vec![g(a("t"), a("u"))]),
    ]
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
        term(r(1), vec![d(c("i"), c("j")), d(v("b"), v("a"))], vec![g(a("u"), a("w")), th(a("x"), a("v"))]),
        term(r(-1), vec![d(c("i"), c("j")), d(v("b"), v("a"))], vec![l2(a("u"), a("x"), a("w"), a("v"))]),
    ]
}

/// Return the Appendix C5 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> VA.
fn c5() -> Expr {
    vec![
        term(r(1), vec![d(c("i"), c("j")), d(v("b"), v("a"))], vec![g(a("u"), a("w")), th(a("x"), a("v"))]),
        term(r(2), vec![d(c("i"), c("j")), d(v("b"), v("a"))], vec![l2(a("u"), a("x"), a("v"), a("w"))]),
    ]
}

/// Return the Appendix C6 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> VV.
fn c6() -> Expr {
    vec![
        term(r(2), vec![d(c("i"), c("j")), d(v("d"), v("b")), d(v("c"), v("a"))], vec![g(a("u"), a("v"))]),
        term(r(-1), vec![d(c("i"), c("j")), d(v("d"), v("a")), d(v("c"), v("b"))], vec![g(a("u"), a("v"))]),
    ]
}

/// Return the Appendix C7 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CC -> AV.
fn c7() -> Expr {
    vec![
        term(r(2), vec![d(v("b"), v("a")), d(c("i"), c("k")), d(c("j"), c("l"))], vec![th(a("v"), a("u"))]),
        term(r(-1), vec![d(v("b"), v("a")), d(c("i"), c("l")), d(c("j"), c("k"))], vec![th(a("v"), a("u"))]),
    ]
}

/// Return the Appendix C8 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CC -> AA.
fn c8() -> Expr {
    vec![
        term(r(1), vec![d(c("i"), c("k")), d(c("j"), c("l"))], vec![th(a("w"), a("u")), th(a("x"), a("v"))]),
        term(q(-1, 2), vec![d(c("i"), c("k")), d(c("j"), c("l"))], vec![th(a("w"), a("v")), th(a("x"), a("u"))]),
        term(r(1), vec![d(c("i"), c("k")), d(c("j"), c("l"))], vec![l2(a("w"), a("x"), a("u"), a("v"))]),

        term(r(1), vec![d(c("i"), c("l")), d(c("j"), c("k"))], vec![th(a("w"), a("v")), th(a("x"), a("u"))]),
        term(q(-1, 2), vec![d(c("i"), c("l")), d(c("j"), c("k"))], vec![th(a("w"), a("u")), th(a("x"), a("v"))]),
        term(r(1), vec![d(c("i"), c("l")), d(c("j"), c("k"))], vec![l2(a("w"), a("x"), a("v"), a("u"))]),
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
        term(q(1, 2), delta.clone(), vec![g(a("u"), a("x")), th(a("y"), a("v")), th(a("z"), a("w"))]),
        term(q(-1, 4), delta.clone(), vec![g(a("u"), a("x")), th(a("y"), a("w")), th(a("z"), a("v"))]),
        term(q(1, 2), delta.clone(), vec![g(a("u"), a("x")), l2(a("y"), a("z"), a("v"), a("w"))]),
        term(r(1), delta.clone(), vec![th(a("y"), a("v")), l2(a("u"), a("z"), a("w"), a("x"))]),
        term(q(-1, 2), delta.clone(), vec![th(a("y"), a("w")), l2(a("u"), a("z"), a("v"), a("x"))]),
        term(q(-1, 2), delta.clone(), vec![th(a("z"), a("v")), l2(a("u"), a("y"), a("w"), a("x"))]),
        term(q(-1, 2), delta.clone(), vec![th(a("z"), a("w")), l2(a("u"), a("y"), a("x"), a("v"))]),
        term(r(-1), delta, vec![l3([a("u"), a("y"), a("z")], [a("w"), a("v"), a("x")])]),
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
        term(q(1, 2), delta.clone(), vec![th(a("z"), a("v")), g(a("t"), a("x")), g(a("u"), a("y"))]),
        term(q(-1, 4), delta.clone(), vec![th(a("z"), a("v")), g(a("t"), a("y")), g(a("u"), a("x"))]),
        term(q(1, 2), delta.clone(), vec![th(a("z"), a("v")), l2(a("t"), a("u"), a("x"), a("y"))]),
        term(q(-1, 2), delta.clone(), vec![g(a("t"), a("x")), l2(a("u"), a("z"), a("y"), a("v"))]),
        term(q(-1, 2), delta.clone(), vec![g(a("t"), a("y")), l2(a("u"), a("z"), a("v"), a("x"))]),
        term(r(1), delta.clone(), vec![g(a("u"), a("y")), l2(a("t"), a("z"), a("v"), a("x"))]),
        term(q(-1, 2), delta.clone(), vec![g(a("u"), a("x")), l2(a("t"), a("z"), a("v"), a("y"))]),
        term(r(1), delta, vec![l3([a("t"), a("u"), a("z")], [a("v"), a("y"), a("x")])]),
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
        term(q(-1, 2), d1.clone(), vec![g(a("t"), a("w")), g(a("u"), a("v"))]),
        term(r(1), d1, vec![l2(a("t"), a("u"), a("v"), a("w"))]),

        term(r(1), d2.clone(), vec![g(a("u"), a("v")), g(a("t"), a("w"))]),
        term(q(-1, 2), d2.clone(), vec![g(a("u"), a("w")), g(a("t"), a("v"))]),
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
        term(r(1), vec![], vec![l4([a("p"), a("r"), a("t"), a("v")], [a("q"), a("s"), a("u"), a("w")])]),

        term(q(1, 2), vec![], vec![th(a("v"), a("s")), l3([a("p"), a("r"), a("t")], [a("q"), a("w"), a("u")])]),
        term(q(1, 2), vec![], vec![th(a("v"), a("q")), l3([a("p"), a("r"), a("t")], [a("w"), a("s"), a("u")])]),
        term(q(1, 2), vec![], vec![th(a("t"), a("s")), l3([a("p"), a("r"), a("v")], [a("q"), a("u"), a("w")])]),
        term(q(1, 2), vec![], vec![th(a("t"), a("q")), l3([a("p"), a("r"), a("v")], [a("u"), a("s"), a("w")])]),

        term(q(-1, 2), vec![], vec![g(a("r"), a("u")), l3([a("p"), a("t"), a("v")], [a("q"), a("s"), a("w")])]),
        term(q(-1, 2), vec![], vec![g(a("r"), a("w")), l3([a("p"), a("t"), a("v")], [a("q"), a("u"), a("s")])]),
        term(q(-1, 2), vec![], vec![g(a("p"), a("u")), l3([a("r"), a("t"), a("v")], [a("s"), a("q"), a("w")])]),
        term(q(-1, 2), vec![], vec![g(a("p"), a("w")), l3([a("r"), a("t"), a("v")], [a("s"), a("u"), a("q")])]),

        term(q(1, 4), vec![], vec![th(a("t"), a("q")), th(a("v"), a("s")), g(a("p"), a("u")), g(a("r"), a("w"))]),
        term(q(-1, 8), vec![], vec![th(a("t"), a("q")), th(a("v"), a("s")), g(a("p"), a("w")), g(a("r"), a("u"))]),
        term(q(1, 4), vec![], vec![th(a("t"), a("q")), th(a("v"), a("s")), l2(a("p"), a("r"), a("u"), a("w"))]),

        term(q(1, 4), vec![], vec![th(a("t"), a("s")), th(a("v"), a("q")), g(a("p"), a("w")), g(a("r"), a("u"))]),
        term(q(-1, 8), vec![], vec![th(a("t"), a("s")), th(a("v"), a("q")), g(a("p"), a("u")), g(a("r"), a("w"))]),
        term(q(1, 4), vec![], vec![th(a("t"), a("s")), th(a("v"), a("q")), l2(a("p"), a("r"), a("w"), a("u"))]),

        term(q(1, 4), vec![], vec![g(a("p"), a("u")), g(a("r"), a("w")), l2(a("t"), a("v"), a("q"), a("s"))]),
        term(q(1, 4), vec![], vec![g(a("p"), a("w")), g(a("r"), a("u")), l2(a("t"), a("v"), a("s"), a("q"))]),

        term(q(1, 3), vec![], vec![l2(a("t"), a("v"), a("q"), a("s")), l2(a("p"), a("r"), a("u"), a("w"))]),
        term(q(1, 3), vec![], vec![l2(a("t"), a("v"), a("s"), a("q")), l2(a("p"), a("r"), a("w"), a("u"))]),
        term(q(1, 6), vec![], vec![l2(a("t"), a("v"), a("q"), a("s")), l2(a("p"), a("r"), a("w"), a("u"))]),
        term(q(1, 6), vec![], vec![l2(a("t"), a("v"), a("s"), a("q")), l2(a("p"), a("r"), a("u"), a("w"))]),

        term(q(1, 2), vec![], vec![th(a("v"), a("s")), g(a("r"), a("w")), l2(a("p"), a("t"), a("q"), a("u"))]),
        term(q(-1, 4), vec![], vec![th(a("v"), a("s")), g(a("r"), a("u")), l2(a("p"), a("t"), a("q"), a("w"))]),
        term(q(-1, 4), vec![], vec![th(a("v"), a("q")), g(a("r"), a("w")), l2(a("p"), a("t"), a("s"), a("u"))]),
        term(q(-1, 4), vec![], vec![th(a("v"), a("q")), g(a("r"), a("u")), l2(a("p"), a("t"), a("w"), a("s"))]),

        term(q(-1, 4), vec![], vec![th(a("t"), a("s")), g(a("r"), a("w")), l2(a("p"), a("v"), a("q"), a("u"))]),
        term(q(1, 2), vec![], vec![th(a("t"), a("s")), g(a("r"), a("u")), l2(a("p"), a("v"), a("q"), a("w"))]),
        term(q(-1, 4), vec![], vec![th(a("t"), a("q")), g(a("r"), a("w")), l2(a("p"), a("v"), a("u"), a("s"))]),
        term(q(-1, 4), vec![], vec![th(a("t"), a("q")), g(a("r"), a("u")), l2(a("p"), a("v"), a("s"), a("w"))]),

        term(q(-1, 4), vec![], vec![th(a("v"), a("s")), g(a("p"), a("w")), l2(a("r"), a("t"), a("q"), a("u"))]),
        term(q(-1, 4), vec![], vec![th(a("v"), a("s")), g(a("p"), a("u")), l2(a("r"), a("t"), a("w"), a("q"))]),
        term(q(1, 2), vec![], vec![th(a("v"), a("q")), g(a("p"), a("w")), l2(a("r"), a("t"), a("s"), a("u"))]),
        term(q(-1, 4), vec![], vec![th(a("v"), a("q")), g(a("p"), a("u")), l2(a("r"), a("t"), a("s"), a("w"))]),

        term(q(-1, 4), vec![], vec![th(a("t"), a("s")), g(a("p"), a("w")), l2(a("r"), a("v"), a("u"), a("q"))]),
        term(q(-1, 4), vec![], vec![th(a("t"), a("s")), g(a("p"), a("u")), l2(a("r"), a("v"), a("q"), a("w"))]),
        term(q(-1, 4), vec![], vec![th(a("t"), a("q")), g(a("p"), a("w")), l2(a("r"), a("v"), a("s"), a("u"))]),
        term(q(1, 2), vec![], vec![th(a("t"), a("q")), g(a("p"), a("u")), l2(a("r"), a("v"), a("s"), a("w"))]),

        term(q(-1, 2), vec![], vec![l2(a("p"), a("r"), a("w"), a("s")), l2(a("t"), a("v"), a("u"), a("q"))]),
        term(q(-1, 2), vec![], vec![l2(a("p"), a("r"), a("u"), a("s")), l2(a("t"), a("v"), a("q"), a("w"))]),
        term(q(-1, 2), vec![], vec![l2(a("p"), a("r"), a("q"), a("w")), l2(a("t"), a("v"), a("u"), a("s"))]),
        term(q(-1, 2), vec![], vec![l2(a("p"), a("r"), a("q"), a("u")), l2(a("t"), a("v"), a("s"), a("w"))]),

        term(q(-1, 2), vec![], vec![l2(a("t"), a("r"), a("q"), a("s")), l2(a("p"), a("v"), a("u"), a("w"))]),
        term(q(-1, 2), vec![], vec![l2(a("t"), a("r"), a("u"), a("w")), l2(a("p"), a("v"), a("q"), a("s"))]),
        term(q(-1, 2), vec![], vec![l2(a("v"), a("r"), a("q"), a("s")), l2(a("t"), a("p"), a("u"), a("w"))]),
        term(q(-1, 2), vec![], vec![l2(a("p"), a("t"), a("q"), a("s")), l2(a("r"), a("v"), a("u"), a("w"))]),

        term(r(1), vec![], vec![l2(a("r"), a("v"), a("s"), a("w")), l2(a("p"), a("t"), a("q"), a("u"))]),
        term(r(1), vec![], vec![l2(a("r"), a("t"), a("s"), a("u")), l2(a("p"), a("v"), a("q"), a("w"))]),

        term(q(-1, 2), vec![], vec![l2(a("r"), a("v"), a("s"), a("u")), l2(a("p"), a("t"), a("q"), a("w"))]),
        term(q(-1, 2), vec![], vec![l2(a("r"), a("v"), a("q"), a("w")), l2(a("p"), a("t"), a("s"), a("u"))]),
        term(q(-1, 2), vec![], vec![l2(a("p"), a("v"), a("q"), a("u")), l2(a("r"), a("t"), a("s"), a("w"))]),
        term(q(-1, 2), vec![], vec![l2(a("r"), a("t"), a("q"), a("u")), l2(a("p"), a("v"), a("s"), a("w"))]),

        term(q(1, 3), vec![], vec![l2(a("r"), a("v"), a("q"), a("u")), l2(a("p"), a("t"), a("s"), a("w"))]),
        term(q(1, 3), vec![], vec![l2(a("r"), a("v"), a("u"), a("q")), l2(a("p"), a("t"), a("w"), a("s"))]),
        term(q(1, 6), vec![], vec![l2(a("r"), a("v"), a("u"), a("q")), l2(a("p"), a("t"), a("s"), a("w"))]),
        term(q(1, 6), vec![], vec![l2(a("r"), a("v"), a("q"), a("u")), l2(a("p"), a("t"), a("w"), a("s"))]),

        term(q(1, 3), vec![], vec![l2(a("r"), a("t"), a("q"), a("w")), l2(a("p"), a("v"), a("s"), a("u"))]),
        term(q(1, 3), vec![], vec![l2(a("r"), a("t"), a("w"), a("q")), l2(a("p"), a("v"), a("u"), a("s"))]),
        term(q(1, 6), vec![], vec![l2(a("r"), a("t"), a("w"), a("q")), l2(a("p"), a("v"), a("s"), a("u"))]),
        term(q(1, 6), vec![], vec![l2(a("r"), a("t"), a("q"), a("w")), l2(a("p"), a("v"), a("u"), a("s"))]),
    ]
}

/// Return the Appendix C13 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> V / AA -> AV.
fn c13() -> Expr {
    vec![
        term(r(1), vec![d(v("b"), v("a"))], vec![l2(a("u"), a("x"), a("w"), a("v"))]),
    ]
}

/// Return the Appendix C14 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for C -> A / CA -> AA.
fn c14() -> Expr {
    vec![
        term(r(-1), vec![d(c("i"), c("j"))], vec![l2(a("w"), a("x"), a("u"), a("v"))]),
    ]
}

/// Return the Appendix C15 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for A -> A / AA -> AA.
fn c15() -> Expr {
    vec![
        term(r(1), vec![], vec![l3([a("t"), a("y"), a("z")], [a("u"), a("w"), a("x")])]),
        term(q(-1, 2), vec![], vec![g(a("t"), a("w")), l2(a("y"), a("z"), a("u"), a("x"))]),
        term(q(-1, 2), vec![], vec![g(a("t"), a("x")), l2(a("y"), a("z"), a("w"), a("u"))]),
        term(q(1, 2), vec![], vec![th(a("y"), a("u")), l2(a("t"), a("z"), a("w"), a("x"))]),
        term(q(1, 2), vec![], vec![th(a("z"), a("u")), l2(a("t"), a("y"), a("x"), a("w"))]),
    ]
}

/// Return the Appendix C16 target expression.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Expr`: Target expression for CA -> AV / CA -> VA.
fn c16() -> Expr {
    vec![
        term(q(-1, 2), vec![d(c("i"), c("j")), d(v("b"), v("a"))], vec![g(a("u"), a("x")), th(a("y"), a("w"))]),
        term(r(-1), vec![d(c("i"), c("j")), d(v("b"), v("a"))], vec![l2(a("u"), a("y"), a("w"), a("x"))]),
    ]
}

/// Return the Appendix C target expression for one metric block.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `Expr`: Target expression.
pub fn block(name: &str) -> Expr {
    match name {
        "C1" => c1(),
        "C2" => c2(),
        "C3" => c3(),
        "C4" => c4(),
        "C5" => c5(),
        "C6" => c6(),
        "C7" => c7(),
        "C8" => c8(),
        "C9" => c9(),
        "C10" => c10(),
        "C11" => c11(),
        "C12" => c12(),
        "C13" => c13(),
        "C14" => c14(),
        "C15" => c15(),
        "C16" => c16(),
        _ => panic!("no Appendix C target for {name}"),
    }
}
