from __future__ import annotations

from fractions import Fraction
from functools import cache
from itertools import combinations, permutations

from core import Expr, Tensor, Term, combine
from spinsum import solveConsistent, spinGram

ZERO_FRACTION = Fraction(0)

@cache
def lowerPermutations(rank: int) -> tuple[tuple[int, ...], ...]:
    """
    Return lower-index permutations for one cumulant rank.

    Notation:
        S_k

    Examples:
        rank 3 returns all elements of S_3 in deterministic order.
    """
    return tuple(permutations(range(rank)))

@cache
def lowerPermutationIndex(rank: int) -> dict[tuple[int, ...], int]:
    """
    Return column indices for lower-index permutations.

    Notation:
        \pi \in S_k \mapsto j

    Examples:
        Used to build spin-projection systems without repeated linear searches.
    """
    return {
        perm: index
        for index, perm in enumerate(lowerPermutations(rank))
    }

def lambdaRank(name: str) -> int | None:
    """
    Return the cumulant rank for high-rank Lambda tensors.

    Notation:
        Lambda3 -> 3
        Lambda4 -> 4

    Examples:
        lambdaRank("Lambda3") gives 3.
        lambdaRank("Lambda4") gives 4.
        lambdaRank("Lambda2") gives None.
    """
    if name == "Lambda3":
        return 3

    if name == "Lambda4":
        return 4

    return None

def canonicalTensor(tensorIn: Tensor) -> Tensor:
    """
    Canonicalise tensor symmetries.

    Notation:
        Lambda2^{pq}_{rs} = Lambda2^{qp}_{sr}.

    Examples:
        Lambda2^{ux}_{vw} and Lambda2^{xu}_{wv} represent the same tensor
        component under simultaneous upper/lower pair exchange.

        This function returns whichever of these two Tensor objects is
        smaller under dataclass ordering.
    """

    # Canonicalise by constructing the swapped form and keeping the smaller one.
    if tensorIn.name == "Lambda2" and len(tensorIn.upper) == 2 and len(tensorIn.lower) == 2:
        swapped = Tensor(
            "Lambda2",
            (tensorIn.upper[1], tensorIn.upper[0]),
            (tensorIn.lower[1], tensorIn.lower[0]),
        )

        return min(tensorIn, swapped)

    return tensorIn

def canonicaliseTensorSymmetry(expr: Expr) -> Expr:
    """
    Canonicalise tensor factors inside every term.

    Notation:
        Apply canonicalTensor(...) to every tensor factor, then combine
        identical scalar products.

    Examples:
        Lambda2^{ux}_{vw} + Lambda2^{xu}_{wv} becomes 2 Lambda2^{canonical}_{canonical}
        where the canonical representative is chosen by dataclass ordering.
    """
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
    """
    Project lower-permutation coefficients into spin-component space.

    Notation:
        v = G c

        where G is the rank-k spin-projection Gram matrix,
        c is a coefficient vector over lower-index permutations.

    Examples:
        For rank = 3 and

            c_{(0,1,2)} = 1
            c_{(0,2,1)} = -1

        Then this function returns the projected spin-vector G c
        as a list of exact Fractions.
    """

    # All lower index permutations.
    perms = lowerPermutations(rank)

    # Spin-projection Gram matrix.
    gram = spinGram(rank)
    out = []
    
    for i, _ in enumerate(perms):
        value = ZERO_FRACTION
    
        # Calculate value_i = \sum_j G_{ij} c_j.
        for j, perm in enumerate(perms):
            value += gram[i][j] * coeffs.get(perm, ZERO_FRACTION)

        out.append(value)

    return out

def sparsestEquivalentOrbit(
    rank: int,
    coeffs: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    """
    Find the sparsest equivalent lower-permutation representative. The input represents 
    a spin-free cumulant combination

            \sum_pi c_\pi Lambda^{p_1 \cdots p_k}_{q_{\pi(1)} \cdots q_{\pi(k)}}

    For Lambda3/Lambda4, as the lower-permutation basis is overcomplete, different coefficient 
    vectors can represent the same projected spin-orbital cumulant. This is important as it means 
    for generated Rust functions there are fewer terms to evaluate, hence less expense.

    Notation:
        Two coefficient vectors c and d are equivalent when

            G c = G d

        where G is the rank-k spin-projection Gram matrix.

    This function searches for a coefficient vector d with the fewest
    non-zero lower-permutation terms such that

            G d = G c.

    Examples:
        For rank 3,

            \Lambda_{3, abc} + \Lambda_{3, acb} + \Lambda_{3, bac} + \Lambda_{3, bca} + \Lambda_{3, cab}

        becomes

            -\Lambda_{3, cba}
    """
    # All lower-index permutations.
    perms = lowerPermutations(rank)
    permIndex = lowerPermutationIndex(rank)
    # Vector G c.
    target = projectedVector(rank, coeffs)
    # Gram matrix for equivalence test.
    gram = spinGram(rank)
    # Remove any zero entries.
    nonzero = {
        perm: coeff
        for perm, coeff in coeffs.items()
        if coeff != 0
    }

    if not nonzero:
        return {}
    
    # Don't find a larger expression.
    currentSupport = len(nonzero)
    
    # Search by increasing expression size and return the first consistent 
    # solution found. This gives a deterministic gauge.
    for supportSize in range(1, currentSupport + 1):
        # Choose subset of lower permutations.
        for support in combinations(perms, supportSize):
            # Build subset linear system G_s x = t (target).
            mat = [
                [gram[row][permIndex[perm]] for perm in support]
                for row in range(len(perms))
            ]
            
            # Attempt to solve.
            try:
                sol = solveConsistent(mat, target)
            except ValueError:
                continue
            
            # Convert into dictionary.
            candidate = {
                perm: coeff
                for perm, coeff in zip(support, sol)
                if coeff != 0
            }
            
            # Return solution if expression is sparser.
            if len(candidate) <= supportSize:
                return candidate
    
    # Otherwise return the original expression.
    return nonzero

@cache
def cachedSparsestEquivalentOrbit(
    rank: int,
    coeffItems: tuple[tuple[tuple[int, ...], Fraction], ...],
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    """
    Return a cached sparsest lower-permutation representative.

    Notation:
        G c = G d

    Examples:
        Many generated terms share the same coefficient vector and reuse one
        solved gauge representative.
    """
    return tuple(
        sorted(
            sparsestEquivalentOrbit(
                rank,
                dict(coeffItems),
            ).items()
        )
    )

def sparsifyHighRankCumulantOrbits(expr: Expr) -> Expr:
    """
    Sparsify Lambda3/Lambda4 lower-index orbit gauges. For each compatible group of 
    terms containing either one \Lambda_3 or \Lambda_4 tensor, collect the coefficients of 
    all lower-index permutations and replace them by the sparsest equivalent representation.

     Notation:
        For fixed surrounding scalar factors,

            \sum_\pi c_\pi Lambda^{p_1 \cdots p_k}_{q_{\pi(1)} \cdots q_{\pi(k)}}

        is replaced by

            \sum_pi d_\pi Lambda^{p_1 \cdots p_k}_{q_{\pi(1)} \cdots q_{\pi(k)}}

        where

            G c = G d

        and d has minimal support found by sparsestEquivalentOrbit(...).
    
    Terms are ground only when they have the same rank, deltas, tensors, \Lambda name,
    \Lambda upper indices, unordered \Lambda lower indices.
    """
    # Put each Lambda inot a coefficient dictionary over lower-index permutations.
    grouped: dict[tuple, dict[tuple[int, ...], Fraction]] = {}
    # Terms to ignore.
    passthrough = []
    
    # High rank terms are 3 or 4.
    for term in expr:
        high = [
            tensorIn
            for tensorIn in term.tensors
            if lambdaRank(tensorIn.name) in (3, 4)
        ]
        
        # Ignore terms with more than one high-rank cumulant.
        if len(high) != 1:
            passthrough.append(term)
            continue
        
        lam = high[0]
        rank = lambdaRank(lam.name)
        
        # This check may not be required.
        if rank is None:
            passthrough.append(term)
            continue
        
        # Check for expected number of indices.
        if len(lam.upper) != rank or len(lam.lower) != rank:
            passthrough.append(term)
            continue
        
        # Repeated lower indices would introduce ambiguity in permutation map 
        # so we ignore them.
        if len(set(lam.lower)) != rank:
            passthrough.append(term)
            continue
        
        # Canonical unordered base for lower indices.
        lowerBase = tuple(sorted(lam.lower))
        # Convert actual lower ordering into a permutation from base.
        lowerPerm = tuple(
            lowerBase.index(idx)
            for idx in lam.lower
        )
        
        # Remove \Lambda tensor from term.
        otherTensors = []
        removed = False
        for tensorIn in term.tensors:
            if tensorIn == lam and not removed:
                removed = True
                continue

            otherTensors.append(tensorIn)
        
        # Group key defining lower-permutation orbit.
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
        
        # Accumulate coefficient for this lower-permutation.
        grouped[key][lowerPerm] = grouped[key].get(lowerPerm, ZERO_FRACTION) + term.coeff

    out = list(passthrough)
    
    # Process each high rank \Lambda.
    for key, coeffs in grouped.items():
        rank, deltas, otherTensors, name, upper, lowerBase = key

        # Replace coeff vector by a sparser equivalent.
        sparse = dict(
            cachedSparsestEquivalentOrbit(
                rank,
                tuple(sorted(coeffs.items())),
            )
        )

        for perm, coeff in sorted(sparse.items()):
            if coeff == 0:
                continue
            
            # Reconstruct tensor.
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

def canonicaliseForOutput(expr: Expr, combined: bool = False) -> Expr:
    """Canonical form for LaTeX/Rust emission.

    Notation:
        Apply only identities that are generally valid:

            Lambda2^{pq}_{rs} = Lambda2^{qp}_{sr}
            G c = G d for Lambda3/Lambda4 spin-free projection gauges

    Examples:
        r2Expr passes combined = True after termsAcc has already been reduced
        to one coefficient per scalar product.
    """
    return sparsifyHighRankCumulantOrbits(
        canonicaliseTensorSymmetry(
            expr if combined else combine(expr)
        )
    )
