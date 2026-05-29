# tools/wick/wick.py

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from symbols import (
    Expr,
    Idx,
    E1,
    Tensor,
    Term,
    Delta,
    combineLikeTerms,
)


@dataclass(frozen = True, order = True)
class NormalOrderedE:
    """A normal-ordered spin-free generator string.

    Represents:

        E^{p1 p2 ... pk}_{q1 q2 ... qk}

    with the spin-free convention:

        E^{p1...pk}_{q1...qk}
        =
        sum_{sigma1...sigmak}
        a^dagger_{p1 sigma1}
        ...
        a^dagger_{pk sigmak}
        a_{qk sigmak}
        ...
        a_{q1 sigma1}

    Its expectation value is:

        <Phi|E^{p1...pk}_{q1...qk}|Phi>
        =
        Gamma^{p1...pk}_{q1...qk}
    """

    upper: tuple[Idx, ...]
    lower: tuple[Idx, ...]

    @property
    def rank(self) -> int:
        return len(self.upper)


@dataclass(frozen = True)
class NormalOrderedTerm:
    """One intermediate term during spin-free generator normal ordering.

    Represents:

        coeff * product(deltas) * E^{p1...pk}_{q1...qk}
    """

    coeff: Fraction
    deltas: tuple[Delta, ...]
    normalOrdered: NormalOrderedE


def normalOrderedTerm(
    normalOrdered: NormalOrderedE,
    *,
    coeff: Fraction = Fraction(1),
    deltas: tuple[Delta, ...] = (),
) -> NormalOrderedTerm | None:
    """Construct a normal-ordering intermediate term.

    Cross-space deltas are removed immediately:

        delta(i, a) = 0
        delta(i, t) = 0
        delta(t, a) = 0
    """

    if coeff == 0:
        return None

    for d in deltas:
        if d.left.space != d.right.space:
            return None

    return NormalOrderedTerm(
        coeff = coeff,
        deltas = deltas,
        normalOrdered = normalOrdered,
    )


def combineNormalOrderedTerms(terms: list[NormalOrderedTerm]) -> list[NormalOrderedTerm]:
    """Combine identical normal-ordering intermediate terms."""

    acc: dict[tuple[tuple[Delta, ...], NormalOrderedE], Fraction] = {}

    for t in terms:
        deltas = tuple(sorted(t.deltas))
        key = (deltas, t.normalOrdered)
        acc[key] = acc.get(key, Fraction(0)) + t.coeff

    out = []

    for (deltas, normalOrdered), coeff in acc.items():
        if coeff != 0:
            out.append(
                NormalOrderedTerm(
                    coeff = coeff,
                    deltas = deltas,
                    normalOrdered = normalOrdered,
                )
            )

    return sorted(
        out,
        key = lambda t: (
            t.normalOrdered.rank,
            tuple(i.name for i in t.normalOrdered.upper),
            tuple(i.name for i in t.normalOrdered.lower),
            tuple((d.left.name, d.right.name) for d in t.deltas),
            t.coeff,
        ),
    )


def leftMultiplyE1IntoNormalOrdered(gen: E1, termIn: NormalOrderedTerm) -> list[NormalOrderedTerm]:
    """Left-multiply a normal-ordered spin-free generator by one E1.

    Implements:

        E^p_q E^{r1...rk}_{s1...sk}
        =
        E^{p r1...rk}_{q s1...sk}
        +
        sum_m delta_{q r_m}
              E^{r1...r_{m-1} p r_{m+1}...rk}_{s1...sk}

    For k = 1:

        E^p_q E^r_s
        =
        E^{pr}_{qs}
        +
        delta_{qr} E^p_s

    Taking the expectation gives:

        <E^p_q E^r_s>
        =
        Gamma^{pr}_{qs}
        +
        delta_{qr} Gamma^p_s
    """

    p = gen.upper
    q = gen.lower

    upper = termIn.normalOrdered.upper
    lower = termIn.normalOrdered.lower

    out: list[NormalOrderedTerm] = []

    noContraction = normalOrderedTerm(
        NormalOrderedE(
            upper = (p,) + upper,
            lower = (q,) + lower,
        ),
        coeff = termIn.coeff,
        deltas = termIn.deltas,
    )

    if noContraction is not None:
        out.append(noContraction)

    for m, rm in enumerate(upper):
        newUpper = upper[:m] + (p,) + upper[m + 1:]
        newDeltas = termIn.deltas + (Delta(q, rm),)

        contracted = normalOrderedTerm(
            NormalOrderedE(
                upper = newUpper,
                lower = lower,
            ),
            coeff = termIn.coeff,
            deltas = newDeltas,
        )

        if contracted is not None:
            out.append(contracted)

    return out


def normalOrderE1Product(generators: tuple[E1, ...]) -> list[NormalOrderedTerm]:
    """Normal-order an ordered product of spin-free one-body generators.

    Input:

        E^{p1}_{q1} E^{p2}_{q2} ... E^{pn}_{qn}

    Output:

        sum of NormalOrderedTerm objects containing normal-ordered spin-free
        generators.

    The algorithm starts from the rightmost generator and left-multiplies the
    remaining generators from right to left.
    """

    if not generators:
        return [
            NormalOrderedTerm(
                coeff = Fraction(1),
                deltas = (),
                normalOrdered = NormalOrderedE(
                    upper = (),
                    lower = (),
                ),
            )
        ]

    last = generators[-1]

    terms = [
        NormalOrderedTerm(
            coeff = Fraction(1),
            deltas = (),
            normalOrdered = NormalOrderedE(
                upper = (last.upper,),
                lower = (last.lower,),
            ),
        )
    ]

    for gen in reversed(generators[:-1]):
        newTerms = []

        for t in terms:
            newTerms.extend(leftMultiplyE1IntoNormalOrdered(gen, t))

        terms = combineNormalOrderedTerms(newTerms)

    return combineNormalOrderedTerms(terms)


def gammaTensorFromNormalOrdered(normalOrdered: NormalOrderedE) -> Tensor | None:
    """Convert a normal-ordered generator to its spin-free RDM tensor.

    Equation:

        <E^{p1...pk}_{q1...qk}>
        =
        Gamma^{p1...pk}_{q1...qk}

    Rank zero corresponds to the scalar identity.
    """

    if normalOrdered.rank == 0:
        return None

    return Tensor(
        name = f"Gamma{normalOrdered.rank}",
        upper = normalOrdered.upper,
        lower = normalOrdered.lower,
    )


def expectation(expr: Expr) -> Expr:
    """Evaluate the expectation value of a symbolic expression.

    Applies:

        < scalar * E1 E2 ... En >
        =
        scalar * < E1 E2 ... En >

    and converts ordered generator products to spin-free RDMs.

    Example:

        <E^p_q E^r_s>
        =
        Gamma^{pr}_{qs}
        +
        delta_{qr} Gamma^p_s
    """

    out: list[Term] = []

    for termIn in expr:
        if not termIn.generators:
            out.append(termIn)
            continue

        normalOrderedTerms = normalOrderE1Product(termIn.generators)

        for normalOrderedTerm in normalOrderedTerms:
            gammaTensor = gammaTensorFromNormalOrdered(normalOrderedTerm.normalOrdered)

            tensors = termIn.tensors

            if gammaTensor is not None:
                tensors = tensors + (gammaTensor,)

            out.append(
                Term(
                    coeff = termIn.coeff * normalOrderedTerm.coeff,
                    deltas = termIn.deltas + normalOrderedTerm.deltas,
                    tensors = tensors,
                    generators = (),
                )
            )

    return combineLikeTerms(tuple(out))
