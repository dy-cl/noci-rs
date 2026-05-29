# symbols.py

from dataclasses import dataclass
from fractions import Fraction
from enum import Enum
from typing import Iterable

class Space(Enum):
    """ Natural orbital space classifications.

    Core:
        Core orbitals denoted i, j, k, l, ... are doubly occupied (up to some tolerance).

    Active:
        Active orbitals denoted t, u, v, w, ... have some fractional occupation.

    Virtual:
        Virtual orbitals denote a, b, c, d, ... are unoccupied (up to some tolerance).
    """
    CORE = "C"
    ACTIVE = "A"
    VIRTUAL = "V"


@dataclass(frozen = True, order = True)
class Idx:
    """Spin-free orbital index.

    Examples:
        i = Idx("i", Space.CORE)
        t = Idx("t", Space.ACTIVE)
        a = Idx("a", Space.VIRTUAL)
    """

    name: str
    space: Space

    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True, order=True)
class Delta:
    """Kronecker delta between two spin-free orbital indices.

    Represents:

        \delta^p_q

    If the two indices belong to different orbital spaces the delta is zero.
    """

    left: Idx
    right: Idx

@dataclass(frozen = True, order = True)
class Tensor:
    """A scalar tensor factor.

    Examples:
        Tensor("Gamma1", (p,), (q,))
            Gamma^p_q

        Tensor("Theta", (p,), (q,))
            Theta^p_q = 2 delta^p_q - Gamma^p_q

        Tensor("Lambda2", (p, q), (r, s))
            Lambda^{pq}_{rs}

        Tensor("Lambda3", (p, q, r), (s, t, u))
            Lambda^{pqr}_{stu}

        Tensor("Lambda4", (p, q, r, s), (t, u, v, w))
            Lambda^{pqrs}_{tuvw}

    These tensors are scalar factors in the generated expression.
    """

    name: str
    upper: tuple[Idx, ...]
    lower: tuple[Idx, ...]

    @property
    def rank(self) -> int:
        return len(self.upper)

@dataclass(frozen=True, order=True)
class E1:
    """A spin-free one-body generator.

    Represents:
        E^p_q = sum_sigma a^dagger_{p sigma} a_{q sigma}
    """

    upper: Idx
    lower: Idx


@dataclass(frozen=True)
class Term:
    """One term in a symbolic expression.

    Represents:

        coeff
        * product(deltas)
        * product(tensors)
        * orderedProduct(generators)

    Examples:

        -1/2 delta(p, q) Gamma^r_s E^t_u E^v_w

        is:

        Term(
            coeff=Fraction(-1, 2),
            deltas=(Delta(p, q),),
            tensors=(Tensor("Gamma1", (r,), (s,)),),
            generators=(E1(t, u), E1(v, w)),
        )
    """

    coeff: Fraction
    deltas: tuple[Delta, ...] = ()
    tensors: tuple[Tensor, ...] = ()
    generators: tuple[E1, ...] = ()

    def isZero(self) -> bool:
        return self.coeff == 0


Expr = tuple[Term, ...]

def asFraction(value: int | Fraction) -> Fraction:
    """Convert an integer or Fraction into a Fraction."""

    if isinstance(value, Fraction):
        return value
    return Fraction(value)

def zero() -> Expr:
    """Return the zero expression."""

    return ()

def one() -> Expr:
    """Return the scalar identity expression."""

    return (Term(Fraction(1)),)


def term(
    coeff: int | Fraction = 1,
    *,
    deltas: Iterable[Delta] = (),
    tensors: Iterable[Tensor] = (),
    generators: Iterable[E1] = (),
) -> Expr:
    """Construct a one-term expression.

    Returns zero if coeff is zero.
    """

    c = asFraction(coeff)

    if c == 0:
        return zero()

    return (
        Term(
            coeff = c,
            deltas = tuple(deltas),
            tensors = tuple(tensors),
            generators = tuple(generators),
        ),
    )

def delta(left: Idx, right: Idx) -> Expr:
    """Construct a Kronecker-delta expression.

    If the two indices are in different orbital spaces, the delta is zero.
    """

    if left.space != right.space:
        return zero()

    return term(deltas = (Delta(left, right),))


def tensor(name: str, upper: Iterable[Idx], lower: Iterable[Idx]) -> Expr:
    """Construct a scalar tensor expression."""

    return term(tensors = (Tensor(name, tuple(upper), tuple(lower)),))


def add(*exprs: Expr) -> Expr:
    """Add several expressions and combine identical terms."""

    terms: list[Term] = []

    for expr in exprs:
        terms.extend(expr)

    return combineLikeTerms(tuple(terms))

def neg(expr: Expr) -> Expr:
    """Return -expr."""

    return scale(expr, Fraction(-1))


def sub(left: Expr, right: Expr) -> Expr:
    """Return left - right."""

    return add(left, neg(right))


def scale(expr: Expr, coeff: int | Fraction) -> Expr:
    """Multiply an expression by a scalar coefficient."""

    c = asFraction(coeff)

    if c == 0:
        return zero()

    out = []

    for t in expr:
        newCoeff = c * t.coeff

        if newCoeff != 0:
            out.append(
                Term(
                    coeff = newCoeff,
                    deltas = t.deltas,
                    tensors = t.tensors,
                    generators = t.generators,
                )
            )

    return tuple(out)

def mul(left: Expr, right: Expr) -> Expr:
    """Multiply two expressions.

    Scalar factors concatenate and commute later through canonicalisation.
    Generator factors concatenate in order and remain non-commuting.

    Examples:

        (A E1 E2) (B E3) = AB E1 E2 E3
    """

    if not left or not right:
        return zero()

    out: list[Term] = []

    for a in left:
        for b in right:
            coeff = a.coeff * b.coeff

            if coeff == 0:
                continue

            out.append(
                Term(
                    coeff = coeff,
                    deltas = a.deltas + b.deltas,
                    tensors = a.tensors + b.tensors,
                    generators = a.generators + b.generators,
                )
            )

    return combineLikeTerms(tuple(out))

def prod(exprs: Iterable[Expr]) -> Expr:
    """Multiply a sequence of expressions."""

    out = one()

    for expr in exprs:
        out = mul(out, expr)

        if not out:
            return zero()

    return out

def canonicalTerm(term: Term) -> Term | None:
    """Canonicalise scalar factors inside a term.
    """

    if term.coeff == 0:
        return None

    for d in term.deltas:
        if d.left.space != d.right.space:
            return None

    return Term(
        coeff = term.coeff,
        deltas = tuple(sorted(term.deltas)),
        tensors = tuple(sorted(term.tensors)),
        generators= term.generators,
    )

def combineLikeTerms(expr: Expr) -> Expr:
    """Combine terms with identical symbolic factors.

    Terms are considered identical if their deltas, tensors, and ordered
    generators match after scalar-factor canonicalisation.
    """

    acc: dict[tuple[tuple[Delta, ...], tuple[Tensor, ...], tuple[E1, ...]], Fraction] = {}

    for t in expr:
        ct = canonicalTerm(t)

        if ct is None:
            continue

        key = (ct.deltas, ct.tensors, ct.generators)
        acc[key] = acc.get(key, Fraction(0)) + ct.coeff

    out = [
        Term(
            coeff = coeff,
            deltas = deltas,
            tensors = tensors,
            generators = generators,
        )
        for (deltas, tensors, generators), coeff in acc.items()
        if coeff != 0
    ]

    return tuple(sorted(out, key = termSortKey))

def simplifyDeltas(expr: Expr) -> Expr:
    """Apply simple delta simplifications.

    Examples:
        delta(i, a) = 0
        delta(i, t) = 0
        delta(t, a) = 0
    """

    out = []

    for t in expr:
        keep = True

        for d in t.deltas:
            if d.left.space != d.right.space:
                keep = False
                break

        if keep:
            out.append(t)

    return combineLikeTerms(tuple(out))


def termSortKey(t: Term) -> tuple:
    """Deterministic ordering key for terms."""

    return (
        len(t.generators),
        tuple((g.upper.name, g.lower.name) for g in t.generators),
        tuple((d.left.name, d.right.name) for d in t.deltas),
        tuple(
            (
                x.name,
                tuple(i.name for i in x.upper),
                tuple(i.name for i in x.lower),
            )
            for x in t.tensors
        ),
        t.coeff,
    )


def exprIsScalar(expr: Expr) -> bool:
    """Return True if no term contains non-commuting generators."""

    return all(len(t.generators) == 0 for t in expr)


def requireScalar(expr: Expr) -> None:
    """Raise if an expression still contains unresolved generators."""

    bad = [t for t in expr if t.generators]

    if bad:
        raise ValueError("Expression still contains unresolved E1 generators.")


def idx(name: str, space: Space) -> Idx:
    """Convenience constructor for indices."""

    return Idx(name = name, space = space)


def core(name: str) -> Idx:
    """Construct a core index."""

    return idx(name, Space.CORE)


def active(name: str) -> Idx:
    """Construct an active index."""

    return idx(name, Space.ACTIVE)


def virtual(name: str) -> Idx:
    """Construct a virtual index."""

    return idx(name, Space.VIRTUAL)
