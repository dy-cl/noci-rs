# tools/wick/generators.py

from fractions import Fraction

from symbols import (
    Expr,
    Idx,
    E1,
    delta,
    term,
    tensor,
    add,
    sub,
    scale,
    mul,
)


def e1(p: Idx, q: Idx) -> Expr:
    """Spin-free one-body generator.

    Represents:
        E^p_q = sum_sigma a^dagger_{p sigma} a_{q sigma}
    """

    return term(generators = (E1(p, q),))


def gamma1(p: Idx, q: Idx) -> Expr:
    """Spin-free one-body RDM Gamma^p_q."""

    return tensor("Gamma1", (p,), (q,))


def theta(p: Idx, q: Idx) -> Expr:
    """Spin-free one-hole RDM Theta^p_q.

    Represents:
        Theta^p_q = 2 delta^p_q - Gamma^p_q
    """

    return tensor("Theta", (p,), (q,))


def lambda2(p: Idx, q: Idx, r: Idx, s: Idx) -> Expr:
    """Spin-free two-body cumulant Lambda^{pq}_{rs}."""

    return tensor("Lambda2", (p, q), (r, s))


def lambda3(p: Idx, q: Idx, r: Idx, s: Idx, t: Idx, u: Idx) -> Expr:
    """Spin-free three-body cumulant Lambda^{pqr}_{stu}."""

    return tensor("Lambda3", (p, q, r), (s, t, u))


def lambda4(
    p: Idx,
    q: Idx,
    r: Idx,
    s: Idx,
    t: Idx,
    u: Idx,
    v: Idx,
    w: Idx,
) -> Expr:
    """Spin-free four-body cumulant Lambda^{pqrs}_{tuvw}."""

    return tensor("Lambda4", (p, q, r, s), (t, u, v, w))


def e2(p: Idx, q: Idx, r: Idx, s: Idx) -> Expr:
    """Spin-free two-body generator.

    Represents:
        E^{pq}_{rs} = E^p_r E^q_s - delta_{rq} E^p_s
    """

    return add(
        mul(e1(p, r), e1(q, s)),
        scale(mul(delta(r, q), e1(p, s)), Fraction(-1)),
    )


def tau1(p: Idx, q: Idx) -> Expr:
    """GNO spin-free single excitation.

    Represents:
        {E^p_q} = E^p_q - Gamma^p_q
    """

    return sub(e1(p, q), gamma1(p, q))


def tau2(p: Idx, q: Idx, r: Idx, s: Idx) -> Expr:
    """GNO spin-free double excitation.

    Represents a two-body GNO operator whose reference expectation vanishes.
    """

    return add(
        e2(p, q, r, s),
        scale(mul(gamma1(p, r), tau1(q, s)), Fraction(-1, 2)),
        scale(mul(gamma1(q, s), tau1(p, r)), Fraction(-1, 2)),
        scale(mul(gamma1(q, r), tau1(p, s)), Fraction(1, 2)),
        scale(mul(gamma1(p, s), tau1(q, r)), Fraction(1, 2)),
        scale(mul(gamma1(p, r), gamma1(q, s)), Fraction(-1)),
        scale(mul(gamma1(p, s), gamma1(q, r)), Fraction(1, 2)),
        scale(lambda2(p, q, r, s), Fraction(-1)),
    )
