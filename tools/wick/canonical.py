from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations

from core import Expr, Tensor, Term, combine
from spinsum import solveConsistent, spinGram


def lambdaRank(name: str) -> int | None:
    if name == "Lambda3":
        return 3

    if name == "Lambda4":
        return 4

    return None


def canonicalTensor(tensorIn: Tensor) -> Tensor:
    """Canonicalise tensor symmetries independent of any block.

    Notation:
        Lambda2^{pq}_{rs} = Lambda2^{qp}_{sr}
    """
    if tensorIn.name == "Lambda2" and len(tensorIn.upper) == 2 and len(tensorIn.lower) == 2:
        swapped = Tensor(
            "Lambda2",
            (tensorIn.upper[1], tensorIn.upper[0]),
            (tensorIn.lower[1], tensorIn.lower[0]),
        )

        return min(tensorIn, swapped)

    return tensorIn


def canonicaliseTensorSymmetry(expr: Expr) -> Expr:
    return combine(tuple(
        Term(
            coeff = term.coeff,
            deltas = term.deltas,
            tensors = tuple(sorted(
                canonicalTensor(tensorIn)
                for tensorIn in term.tensors
            )),
        )
        for term in expr
    ))


def projectedVector(rank: int, coeffs: dict[tuple[int, ...], Fraction]) -> list[Fraction]:
    perms = tuple(permutations(range(rank)))
    gram = spinGram(rank)
    out = []

    for i, _ in enumerate(perms):
        value = Fraction(0)

        for j, perm in enumerate(perms):
            value += gram[i][j] * coeffs.get(perm, Fraction(0))

        out.append(value)

    return out


def sparsestEquivalentOrbit(
    rank: int,
    coeffs: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    """Find the sparsest equivalent lower-permutation representative.

    Notation:
        Two coefficient vectors c and d are equivalent when

            G c = G d

        where G is the rank-k spin-projection Gram matrix.

    Examples:
        For rank 3,

            Lambda3_{abc} + Lambda3_{acb} + Lambda3_{bac}
            + Lambda3_{bca} + Lambda3_{cab}

        becomes

            -Lambda3_{cba}

        if that is the sparsest equivalent representative.
    """
    perms = tuple(permutations(range(rank)))
    target = projectedVector(rank, coeffs)
    gram = spinGram(rank)

    nonzero = {
        perm: coeff
        for perm, coeff in coeffs.items()
        if coeff != 0
    }

    if not nonzero:
        return {}

    currentSupport = len(nonzero)

    for supportSize in range(1, currentSupport + 1):
        for support in combinations(perms, supportSize):
            mat = [
                [gram[row][perms.index(perm)] for perm in support]
                for row in range(len(perms))
            ]

            try:
                sol = solveConsistent(mat, target)
            except ValueError:
                continue

            candidate = {
                perm: coeff
                for perm, coeff in zip(support, sol)
                if coeff != 0
            }

            if len(candidate) <= supportSize:
                return candidate

    return nonzero


def sparsifyHighRankCumulantOrbits(expr: Expr) -> Expr:
    """Sparsify Lambda3/Lambda4 lower-index orbit gauges.

    Notation:
        For each fixed upper tuple, lower-index set, and surrounding scalar
        factors, choose the fewest-term equivalent Lambda3/Lambda4 expression.
    """
    grouped: dict[tuple, dict[tuple[int, ...], Fraction]] = {}
    passthrough = []

    for term in expr:
        high = [
            tensorIn
            for tensorIn in term.tensors
            if lambdaRank(tensorIn.name) in (3, 4)
        ]

        if len(high) != 1:
            passthrough.append(term)
            continue

        lam = high[0]
        rank = lambdaRank(lam.name)

        if rank is None:
            passthrough.append(term)
            continue

        if len(lam.upper) != rank or len(lam.lower) != rank:
            passthrough.append(term)
            continue

        if len(set(lam.lower)) != rank:
            passthrough.append(term)
            continue

        lowerBase = tuple(sorted(lam.lower))
        lowerPerm = tuple(
            lowerBase.index(idx)
            for idx in lam.lower
        )

        otherTensors = []
        removed = False

        for tensorIn in term.tensors:
            if tensorIn == lam and not removed:
                removed = True
                continue

            otherTensors.append(tensorIn)

        key = (
            rank,
            term.deltas,
            tuple(otherTensors),
            lam.name,
            lam.upper,
            lowerBase,
        )

        if key not in grouped:
            grouped[key] = {}

        grouped[key][lowerPerm] = grouped[key].get(lowerPerm, Fraction(0)) + term.coeff

    out = list(passthrough)

    for key, coeffs in grouped.items():
        rank, deltas, otherTensors, name, upper, lowerBase = key
        sparse = sparsestEquivalentOrbit(rank, coeffs)

        for perm, coeff in sorted(sparse.items()):
            if coeff == 0:
                continue

            out.append(
                Term(
                    coeff = coeff,
                    deltas = deltas,
                    tensors = tuple(sorted(
                        otherTensors
                        + (
                            Tensor(
                                name,
                                upper,
                                tuple(lowerBase[i] for i in perm),
                            ),
                        )
                    )),
                )
            )

    return combine(tuple(out))


def canonicaliseForOutput(expr: Expr) -> Expr:
    """Canonical form for LaTeX/Rust emission.

    Notation:
        Apply only identities that are generally valid:

            Lambda2^{pq}_{rs} = Lambda2^{qp}_{sr}
            G c = G d for Lambda3/Lambda4 spin-free projection gauges
    """
    return sparsifyHighRankCumulantOrbits(
        canonicaliseTensorSymmetry(
            combine(expr)
        )
    )
