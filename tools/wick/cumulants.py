# tools/wick/cumulants.py

from __future__ import annotations

from functools import cache
from fractions import Fraction
from itertools import permutations, product as cartesianProduct

from symbols import (
    Expr,
    Space,
    Tensor,
    Term,
    tensor,
    delta,
    term,
    zero,
    add,
    mul,
    prod,
    scale,
    combineLikeTerms,
)


ALPHA = 0
BETA = 1


def permutationSign(mapping: tuple[int, ...]) -> int:
    """Return the sign of a permutation."""
    nInversions = 0

    for i in range(len(mapping)):
        for j in range(i + 1, len(mapping)):
            if mapping[i] > mapping[j]:
                nInversions += 1

    return -1 if nInversions % 2 else 1


def lowerBlockAssignments(
    upperBlocks: tuple[tuple[int, ...], ...],
    rank: int,
) -> list[tuple[tuple[int, ...], ...]]:
    """Return lower-index block assignments matching the upper block sizes."""
    def rec(
        blockIndex: int,
        remaining: tuple[int, ...],
        current: tuple[tuple[int, ...], ...],
    ) -> list[tuple[tuple[int, ...], ...]]:
        if blockIndex == len(upperBlocks):
            return [current]

        blockSize = len(upperBlocks[blockIndex])
        out = []

        for choice in permutations(remaining, blockSize):
            if tuple(sorted(choice)) != choice:
                continue

            choiceSet = set(choice)
            nextRemaining = tuple(i for i in remaining if i not in choiceSet)
            out.extend(
                rec(
                    blockIndex + 1,
                    nextRemaining,
                    current + (choice,),
                )
            )

        return out

    return rec(
        0,
        tuple(range(rank)),
        (),
    )


def gamma2ToLambda(t: Tensor) -> Expr:
    """Rewrite a spin-free two-body RDM into cumulants."""
    if t.name != "Gamma2":
        raise ValueError(f"Expected Gamma2, got {t.name}")

    p, q = t.upper
    r, s = t.lower

    return add(
        mul(
            tensor("Gamma1", (p,), (r,)),
            tensor("Gamma1", (q,), (s,)),
        ),
        scale(
            mul(
                tensor("Gamma1", (p,), (s,)),
                tensor("Gamma1", (q,), (r,)),
            ),
            Fraction(-1, 2),
        ),
        tensor("Lambda2", (p, q), (r, s)),
    )


def gamma1SpinComponent(
    upper,
    lower,
    upperSpin: int,
    lowerSpin: int,
) -> Expr:
    """Return one spin-orbital Gamma1 component in spin-free form."""
    if upperSpin != lowerSpin:
        return zero()

    return scale(
        tensor("Gamma1", (upper,), (lower,)),
        Fraction(1, 2),
    )


def lambda2SpinComponent(
    upper: tuple,
    lower: tuple,
    upperSpins: tuple[int, int],
    lowerSpins: tuple[int, int],
) -> Expr:
    """Return one spin-orbital Lambda2 component in spin-free form."""
    p, q = upper
    r, s = lower
    sp, sq = upperSpins
    sr, ss = lowerSpins
    sign = Fraction(1)

    if (sp, sq) == (BETA, ALPHA):
        p, q = q, p
        sp, sq = sq, sp
        sign *= -1

    if (sp, sq) in {
        (ALPHA, ALPHA),
        (BETA, BETA),
    }:
        if (sr, ss) != (sp, sq):
            return zero()

        return scale(
            add(
                tensor("Lambda2", (p, q), (r, s)),
                scale(
                    tensor("Lambda2", (p, q), (s, r)),
                    Fraction(-1),
                ),
            ),
            sign * Fraction(1, 6),
        )

    if (sp, sq) != (ALPHA, BETA):
        return zero()

    if (sr, ss) == (ALPHA, BETA):
        return scale(
            add(
                scale(
                    tensor("Lambda2", (p, q), (r, s)),
                    2,
                ),
                tensor("Lambda2", (p, q), (s, r)),
            ),
            sign * Fraction(1, 6),
        )

    if (sr, ss) == (BETA, ALPHA):
        return scale(
            add(
                tensor("Lambda2", (p, q), (r, s)),
                scale(
                    tensor("Lambda2", (p, q), (s, r)),
                    2,
                ),
            ),
            sign * Fraction(-1, 6),
        )

    return zero()


def invertPermutation(mapping: tuple[int, ...]) -> tuple[int, ...]:
    """Return the inverse of a permutation."""
    out = [0] * len(mapping)

    for i, j in enumerate(mapping):
        out[j] = i

    return tuple(out)


def composePermutation(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    """Return left o right."""
    return tuple(left[right[i]] for i in range(len(left)))


def cycleCount(mapping: tuple[int, ...]) -> int:
    """Return the number of cycles in a permutation."""
    seen = [False] * len(mapping)
    count = 0

    for i in range(len(mapping)):
        if seen[i]:
            continue

        count += 1
        j = i

        while not seen[j]:
            seen[j] = True
            j = mapping[j]

    return count


def invertFractionMatrix(matrix: tuple[tuple[Fraction, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    """Invert a small rational matrix by Gauss-Jordan elimination."""
    n = len(matrix)
    work = [
        list(row) + [Fraction(1 if i == j else 0) for j in range(n)]
        for i, row in enumerate(matrix)
    ]

    for col in range(n):
        pivot = None

        for row in range(col, n):
            if work[row][col] != 0:
                pivot = row
                break

        if pivot is None:
            raise ValueError("singular spin-replacement Gram matrix")

        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]

        pivotValue = work[col][col]
        work[col] = [value / pivotValue for value in work[col]]

        for row in range(n):
            if row == col:
                continue

            factor = work[row][col]

            if factor == 0:
                continue

            work[row] = [
                value - factor * pivotValue
                for value, pivotValue in zip(work[row], work[col])
            ]

    return tuple(tuple(row[n:]) for row in work)


def rankThreeSpinReplacementInverse(
    gram: tuple[tuple[Fraction, ...], ...],
    perms: tuple[tuple[int, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Return the rank-three inverse on the non-null spin subspace."""
    signs = tuple(Fraction(permutationSign(p)) for p in perms)
    n = len(perms)
    projector = tuple(
        tuple(signs[i] * signs[j] / n for j in range(n))
        for i in range(n)
    )
    shifted = tuple(
        tuple(gram[i][j] + projector[i][j] for j in range(n))
        for i in range(n)
    )
    shiftedInverse = invertFractionMatrix(shifted)

    return tuple(
        tuple(shiftedInverse[i][j] - projector[i][j] for j in range(n))
        for i in range(n)
    )


@cache
def spinReplacementData(rank: int) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[Fraction, ...], ...]]:
    """Return permutation basis and inverse Gram matrix for spin replacement.

    The Gram matrix between permutation-coupled spin strings is
    2^{cycles(pi^{-1} rho)}. Its inverse gives generic spin-replacement
    coefficients. Rank two reproduces Eqs. 25--27 of the paper; rank three
    supplies active Lambda3 replacement with the spin-null sector projected out.
    """
    perms = tuple(permutations(range(rank)))
    gram = []

    for left in perms:
        row = []

        for right in perms:
            rel = composePermutation(invertPermutation(left), right)
            row.append(Fraction(2 ** cycleCount(rel)))

        gram.append(tuple(row))

    gram = tuple(gram)

    if rank == 3:
        return perms, rankThreeSpinReplacementInverse(gram, perms)

    return perms, invertFractionMatrix(gram)


def spinFreeCumulantSpinComponent(
    name: str,
    upper: tuple,
    lower: tuple,
    upperSpins: tuple[int, ...],
    lowerSpins: tuple[int, ...],
) -> Expr:
    """Return one spin-orbital cumulant component in spin-free tensors."""
    rank = len(upper)

    if len(lower) != rank or len(upperSpins) != rank or len(lowerSpins) != rank:
        raise ValueError("inconsistent cumulant spin-component rank")

    perms, inverseGram = spinReplacementData(rank)
    out = zero()

    for row, spinPermutation in enumerate(perms):
        spinMatches = all(
            lowerSpins[pos] == upperSpins[spinPermutation[pos]]
            for pos in range(rank)
        )

        if not spinMatches:
            continue

        for col, lowerPermutation in enumerate(perms):
            coeff = inverseGram[row][col] * permutationSign(lowerPermutation)

            if coeff == 0:
                continue

            out = add(
                out,
                scale(
                    tensor(
                        name,
                        upper,
                        tuple(lower[i] for i in lowerPermutation),
                    ),
                    coeff,
                ),
            )

    out = combineLikeTerms(out)

    if rank == 3 and name == "Lambda3":
        out = removeRankThreeSpinNullAverage(
            out,
            upper,
            lower,
        )

    return out


def removeRankThreeSpinNullAverage(
    expr: Expr,
    upper: tuple,
    lower: tuple,
) -> Expr:
    """Choose a rank-three spin-replacement gauge for the null spin sector."""
    perms = tuple(permutations(range(3)))
    coeffs = {perm: Fraction(0) for perm in perms}

    for termIn in expr:
        if termIn.deltas or len(termIn.tensors) != 1:
            return expr

        t = termIn.tensors[0]

        if t.name != "Lambda3" or t.upper != upper:
            return expr

        try:
            perm = tuple(lower.index(i) for i in t.lower)
        except ValueError:
            return expr

        if tuple(lower[i] for i in perm) != t.lower:
            return expr

        coeffs[perm] += termIn.coeff

    counts: dict[Fraction, int] = {}

    for coeff in coeffs.values():
        counts[coeff] = counts.get(coeff, 0) + 1

    common = [
        coeff
        for coeff, count in counts.items()
        if count == len(perms) - 1
    ]

    if len(common) != 1:
        return expr

    nullCoeff = common[0]
    coeffs = {
        perm: coeff - nullCoeff
        for perm, coeff in coeffs.items()
    }
    out = zero()

    for perm, coeff in coeffs.items():
        if coeff == 0:
            continue

        out = add(
            out,
            scale(
                tensor(
                    "Lambda3",
                    upper,
                    tuple(lower[i] for i in perm),
                ),
                coeff,
            ),
        )

    return combineLikeTerms(out)


def lambda3SpinComponent(
    upper: tuple,
    lower: tuple,
    upperSpins: tuple[int, int, int],
    lowerSpins: tuple[int, int, int],
) -> Expr:
    """Return one spin-orbital Lambda3 component in spin-free form."""
    return spinFreeCumulantSpinComponent(
        "Lambda3",
        upper,
        lower,
        upperSpins,
        lowerSpins,
    )


def cumulantSpinComponent(
    upper: tuple,
    lower: tuple,
    upperSpins: tuple[int, ...],
    lowerSpins: tuple[int, ...],
) -> Expr:
    """Return a connected spin-orbital cumulant component."""
    rank = len(upper)

    if rank == 1:
        return gamma1SpinComponent(
            upper[0],
            lower[0],
            upperSpins[0],
            lowerSpins[0],
        )

    if rank == 2:
        return lambda2SpinComponent(
            upper,
            lower,
            upperSpins,
            lowerSpins,
        )

    if rank == 3:
        return lambda3SpinComponent(
            upper,
            lower,
            upperSpins,
            lowerSpins,
        )

    if rank == 4:
        return lambda4SpinComponent(
            upper,
            lower,
            upperSpins,
            lowerSpins,
        )

    raise NotImplementedError(f"rank-{rank} spin cumulant is unsupported")


def lowerBlockAssignmentsFromPermutation(
    upperBlocks: tuple[tuple[int, ...], ...],
    rank: int,
):
    """Yield lower-index block assignments matching upper-block sizes."""
    seen = set()

    for mapping in permutations(range(rank)):
        out = []
        offset = 0

        for upperBlock in upperBlocks:
            size = len(upperBlock)
            out.append(tuple(sorted(mapping[offset:offset + size])))
            offset += size

        key = tuple(out)

        if key in seen:
            continue

        seen.add(key)
        yield key


def gammaSpinComponent(
    upper: tuple,
    lower: tuple,
    upperSpins: tuple[int, ...],
    lowerSpins: tuple[int, ...],
) -> Expr:
    """Return one spin-orbital RDM component in spin-free cumulants."""
    rank = len(upper)
    positions = tuple(range(rank))
    out = zero()

    for upperBlocks in setPartitions(positions):
        upperSeq = tuple(i for upperBlock in upperBlocks for i in upperBlock)

        for lowerBlocks in lowerBlockAssignmentsFromPermutation(upperBlocks, rank):
            lowerSeq = tuple(i for lowerBlock in lowerBlocks for i in lowerBlock)
            sign = Fraction(
                permutationSign(upperSeq)
                * permutationSign(lowerSeq)
            )
            factors = []

            for upperBlock, lowerBlock in zip(upperBlocks, lowerBlocks):
                factors.append(
                    cumulantSpinComponent(
                        tuple(upper[i] for i in upperBlock),
                        tuple(lower[i] for i in lowerBlock),
                        tuple(upperSpins[i] for i in upperBlock),
                        tuple(lowerSpins[i] for i in lowerBlock),
                    )
                )

            out = add(
                out,
                scale(
                    prod(factors),
                    sign,
                ),
            )

    return combineLikeTerms(out)


def lambda4SpinComponent(
    upper: tuple,
    lower: tuple,
    upperSpins: tuple[int, int, int, int],
    lowerSpins: tuple[int, int, int, int],
) -> Expr:
    """Return one all-active spin-orbital Lambda4 component in spin-free form."""
    raise NotImplementedError(
        "All-active spin-orbital Lambda4 components are not implemented; C12 is unsupported."
    )


def spinOrbitalCumulantBlockFactor(
    t: Tensor,
    upperBlock: tuple[int, ...],
    lowerBlock: tuple[int, ...],
    upperSpins: tuple[int, ...],
    lowerSpins: tuple[int, ...],
) -> Expr:
    """Return one spin-orbital cumulant block in spin-free form.

    This conservative Gamma-route helper only supports rank-one and rank-two
    spin-replacement rules. Mixed higher connected cumulants are evaluated by
    the grouped GNO engine in gno.py, not by patching Gamma4 here.
    """
    upper = tuple(t.upper[i] for i in upperBlock)
    lower = tuple(t.lower[i] for i in lowerBlock)
    blockUpperSpins = tuple(upperSpins[i] for i in upperBlock)
    blockLowerSpins = tuple(lowerSpins[i] for i in lowerBlock)
    rank = len(upper)

    if rank == 1:
        return gamma1SpinComponent(
            upper[0],
            lower[0],
            blockUpperSpins[0],
            blockLowerSpins[0],
        )

    if rank == 2:
        return lambda2SpinComponent(
            upper,
            lower,
            blockUpperSpins,
            blockLowerSpins,
        )

    if rank == 3:
        if all(i.space == Space.ACTIVE for i in upper + lower):
            raise NotImplementedError(
                "All-active spin-orbital Lambda3 components are not implemented in Gamma4 products."
            )

        return zero()

    if rank == 4:
        if all(i.space == Space.ACTIVE for i in upper + lower):
            return lambda4SpinComponent(
                upper,
                lower,
                blockUpperSpins,
                blockLowerSpins,
            )

        return zero()

    raise ValueError(f"Unsupported spin-orbital cumulant rank {rank}")


def gammaDisconnectedTerm(
    t: Tensor,
    upperBlocks: tuple[tuple[int, ...], ...],
    lowerBlocks: tuple[tuple[int, ...], ...],
    rank: int,
) -> Expr:
    """Return one spin-summed disconnected Gamma contribution."""
    upperSeq = tuple(
        i
        for upperBlock in upperBlocks
        for i in upperBlock
    )
    lowerSeq = tuple(
        i
        for lowerBlock in lowerBlocks
        for i in lowerBlock
    )
    sign = Fraction(
        permutationSign(upperSeq)
        * permutationSign(lowerSeq)
    )
    out = zero()

    for spins in cartesianProduct((ALPHA, BETA), repeat = rank):
        factors = []

        for upperBlock, lowerBlock in zip(upperBlocks, lowerBlocks):
            factors.append(
                spinOrbitalCumulantBlockFactor(
                    t,
                    upperBlock,
                    lowerBlock,
                    spins,
                    spins,
                )
            )

        out = add(
            out,
            scale(
                prod(factors),
                sign,
            ),
        )

    return combineLikeTerms(out)


def rank3PermutationCoeff(mapping: tuple[int, ...]) -> Fraction:
    """Return the rank-three spin-free disconnected-product coefficient."""
    if mapping == (0, 1, 2):
        scaleOut = Fraction(1)
    elif mapping in {
        (0, 2, 1),
        (1, 0, 2),
        (2, 1, 0),
    }:
        scaleOut = Fraction(1, 2)
    elif mapping in {
        (1, 2, 0),
        (2, 0, 1),
    }:
        scaleOut = Fraction(1, 4)
    else:
        raise ValueError(f"Unsupported rank-three permutation {mapping}")

    return permutationSign(mapping) * scaleOut


def gamma3BlockFactor(
    t: Tensor,
    upperBlock: tuple[int, ...],
    lowerBlock: tuple[int, ...],
) -> Expr:
    """Return one rank-three disconnected cumulant factor."""
    upper = tuple(t.upper[i] for i in upperBlock)
    lower = tuple(t.lower[i] for i in lowerBlock)
    rank = len(upper)

    if rank == 1:
        return tensor("Gamma1", upper, lower)

    if rank == 2:
        return tensor("Lambda2", upper, lower)

    raise ValueError(f"Unsupported Gamma3 disconnected block rank {rank}")


def gamma3ConnectedContribution(t: Tensor) -> Expr:
    """Return the connected rank-three spin-free cumulant contribution.

    This is the validated all-active Gamma3 reconstruction route.
    Mixed C/A/V GNO products are handled by the grouped GNO evaluator.
    """
    if not allActive(t):
        return zero()

    p, q, r = t.upper
    s, u, v = t.lower

    return add(
        tensor("Lambda3", (p, q, r), (s, u, v)),
        scale(mul(tensor("Gamma1", (p,), (u,)), tensor("Lambda2", (q, r), (s, v))), Fraction(-1, 2)),
        scale(mul(tensor("Gamma1", (p,), (v,)), tensor("Lambda2", (q, r), (u, s))), Fraction(-1, 2)),
        scale(mul(tensor("Theta", (q,), (s,)), tensor("Lambda2", (p, r), (u, v))), Fraction(1, 2)),
        scale(mul(tensor("Theta", (r,), (s,)), tensor("Lambda2", (p, q), (v, u))), Fraction(1, 2)),
        scale(mul(tensor("Gamma1", (p,), (s,)), tensor("Lambda2", (q, r), (v, u))), Fraction(1, 2)),
        scale(mul(tensor("Gamma1", (p,), (u,)), tensor("Lambda2", (q, r), (s, v))), Fraction(1, 2)),
        scale(mul(tensor("Gamma1", (p,), (u,)), tensor("Lambda2", (q, r), (v, s))), Fraction(-1, 4)),
        scale(mul(mul(tensor("Gamma1", (p,), (v,)), tensor("Gamma1", (q,), (u,))), tensor("Theta", (r,), (s,))), Fraction(-1, 4)),
        scale(mul(tensor("Gamma1", (p,), (v,)), tensor("Lambda2", (q, r), (s, u))), Fraction(-1, 4)),
        scale(mul(tensor("Gamma1", (p,), (v,)), tensor("Lambda2", (q, r), (u, s))), Fraction(1, 2)),
        scale(mul(tensor("Gamma1", (q,), (s,)), tensor("Lambda2", (p, r), (v, u))), Fraction(-1, 4)),
        scale(mul(tensor("Gamma1", (q,), (u,)), tensor("Lambda2", (p, r), (s, v))), Fraction(-1, 2)),
        scale(mul(tensor("Gamma1", (q,), (u,)), tensor("Lambda2", (p, r), (v, s))), Fraction(1, 2)),
        scale(mul(tensor("Gamma1", (q,), (v,)), tensor("Lambda2", (p, r), (u, s))), Fraction(-1, 4)),
        scale(mul(tensor("Gamma1", (r,), (s,)), tensor("Lambda2", (p, q), (u, v))), Fraction(-1, 4)),
        scale(mul(tensor("Gamma1", (r,), (u,)), tensor("Lambda2", (p, q), (v, s))), Fraction(-1, 4)),
        scale(mul(tensor("Gamma1", (r,), (v,)), tensor("Lambda2", (p, q), (s, u))), Fraction(-1, 2)),
        scale(mul(tensor("Gamma1", (r,), (v,)), tensor("Lambda2", (p, q), (u, s))), Fraction(1, 2)),
        scale(mul(tensor("Lambda2", (p, q), (v, u)), tensor("Theta", (r,), (s,))), Fraction(-1, 2)),
        scale(mul(tensor("Lambda2", (p, r), (u, v)), tensor("Theta", (q,), (s,))), Fraction(-1, 2)),
    )


def gamma3ToLambda(t: Tensor) -> Expr:
    """Rewrite a spin-free three-body RDM into cumulants."""
    if t.name != "Gamma3":
        raise ValueError(f"Expected Gamma3, got {t.name}")

    positions = (0, 1, 2)
    out = gamma3ConnectedContribution(t)

    for upperBlocks in setPartitions(positions):
        if len(upperBlocks) == 1:
            continue

        for mapping in permutations(positions):
            mapping = tuple(mapping)
            factors = []

            for upperBlock in upperBlocks:
                lowerBlock = tuple(mapping[i] for i in upperBlock)
                factors.append(
                    gamma3BlockFactor(
                        t,
                        upperBlock,
                        lowerBlock,
                    )
                )

            out = add(
                out,
                scale(
                    prod(factors),
                    rank3PermutationCoeff(mapping),
                ),
            )

    return combineLikeTerms(out)


def gamma4BySpinSummation(t: Tensor) -> Expr:
    """Rewrite Gamma4 by conservative spin-summed disconnected products."""
    if allActive(t):
        raise NotImplementedError(
            "All-active spin-free Gamma4/C12 is unsupported in this generator."
        )

    positions = (0, 1, 2, 3)
    out = zero()

    for upperBlocks in setPartitions(positions):
        if len(upperBlocks) == 1:
            continue

        for lowerBlocks in lowerBlockAssignments(
            upperBlocks,
            4,
        ):
            out = add(
                out,
                gammaDisconnectedTerm(
                    t,
                    upperBlocks,
                    lowerBlocks,
                    4,
                ),
            )

    return combineLikeTerms(out)


def setPartitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    """Return all set partitions of an index-position tuple."""
    if not items:
        return [()]

    first = items[0]
    rest = items[1:]
    out: set[tuple[tuple[int, ...], ...]] = set()

    for partition in setPartitions(rest):
        out.add(
            tuple(
                sorted(
                    ((first,),) + partition,
                    key = lambda block: block[0],
                )
            )
        )

        for i, block in enumerate(partition):
            newPartition = list(partition)
            newPartition[i] = tuple(sorted((first,) + block))

            out.add(
                tuple(
                    sorted(
                        newPartition,
                        key = lambda block: block[0],
                    )
                )
            )

    return sorted(
        out,
        key = lambda partition: (
            len(partition),
            partition,
        ),
    )


def gamma4ToLambda(t: Tensor) -> Expr:
    """Rewrite a spin-free four-body RDM into cumulants."""
    if t.name != "Gamma4":
        raise ValueError(f"Expected Gamma4, got {t.name}")

    return gamma4BySpinSummation(t)


def gammaToLambda(t: Tensor) -> Expr:
    """Rewrite one spin-free RDM tensor into cumulant language."""
    if t.name == "Gamma1":
        return tensor("Gamma1", t.upper, t.lower)

    if t.name == "Gamma2":
        return gamma2ToLambda(t)

    if t.name == "Gamma3":
        return gamma3ToLambda(t)

    if t.name == "Gamma4":
        return gamma4ToLambda(t)

    return tensor(t.name, t.upper, t.lower)


def removeOneTensor(tensors: tuple[Tensor, ...], target: Tensor) -> tuple[Tensor, ...] | None:
    """Remove one matching tensor from a tensor tuple."""
    out = list(tensors)

    for i, t in enumerate(out):
        if t == target:
            del out[i]
            return tuple(sorted(out))

    return None


def removeTensorPair(
    tensors: tuple[Tensor, ...],
    first: Tensor,
    second: Tensor,
) -> tuple[Tensor, ...] | None:
    """Remove two matching tensors from a tensor tuple."""
    once = removeOneTensor(tensors, first)

    if once is None:
        return None

    return removeOneTensor(once, second)


def canonicaliseLambda2Tensor(t: Tensor) -> Tensor:
    """Canonicalise Lambda2 pair-exchange symmetry."""
    if t.name != "Lambda2":
        return t

    p, q = t.upper
    r, s = t.lower
    swapped = Tensor(
        name = "Lambda2",
        upper = (q, p),
        lower = (s, r),
    )

    return min(t, swapped)


def canonicaliseTensorSymmetry(expr: Expr) -> Expr:
    """Canonicalise tensor factors using spin-free tensor symmetries."""
    out = []

    for termIn in expr:
        out.append(
            Term(
                coeff = termIn.coeff,
                deltas = termIn.deltas,
                tensors = tuple(
                    canonicaliseLambda2Tensor(t)
                    for t in termIn.tensors
                ),
                generators = termIn.generators,
            )
        )

    return combineLikeTerms(tuple(out))


def lambda3NullKey(termIn: Term, tensorIndex: int) -> tuple | None:
    """Return a grouping key for rank-three spin null-sum simplification."""
    t = termIn.tensors[tensorIndex]

    if t.name != "Lambda3":
        return None

    lowerSet = tuple(sorted(t.lower))

    if len(set(lowerSet)) != 3:
        return None

    return (
        termIn.deltas,
        termIn.tensors[:tensorIndex] + termIn.tensors[tensorIndex + 1:],
        t.upper,
        lowerSet,
        termIn.generators,
    )


def canonicaliseLambda3NullSums(expr: Expr) -> Expr:
    """Remove uniform rank-three Lambda3 null-sum components."""
    terms = list(combineLikeTerms(expr))
    groups: dict[tuple, list[int]] = {}

    for i, termIn in enumerate(terms):
        lambdaPositions = [
            j
            for j, t in enumerate(termIn.tensors)
            if t.name == "Lambda3"
        ]

        if len(lambdaPositions) != 1:
            continue

        key = lambda3NullKey(
            termIn,
            lambdaPositions[0],
        )

        if key is None:
            continue

        groups.setdefault(key, []).append(i)

    adjustments = {i: Fraction(0) for i in range(len(terms))}

    for indices in groups.values():
        if len(indices) != 6:
            continue

        counts: dict[Fraction, int] = {}

        for i in indices:
            coeff = terms[i].coeff
            counts[coeff] = counts.get(coeff, 0) + 1

        common = [
            coeff
            for coeff, count in counts.items()
            if count == 5
        ]

        if len(common) != 1:
            continue

        for i in indices:
            adjustments[i] -= common[0]

    out = []

    for i, termIn in enumerate(terms):
        coeff = termIn.coeff + adjustments[i]

        if coeff == 0:
            continue

        out.append(
            Term(
                coeff = coeff,
                deltas = termIn.deltas,
                tensors = termIn.tensors,
                generators = termIn.generators,
            )
        )

    return combineLikeTerms(tuple(out))


def canonicaliseSpinFreeCumulantsOnce(expr: Expr) -> Expr:
    """Apply one pass of conservative spin-free Lambda2 identities."""
    terms = list(combineLikeTerms(expr))
    used: set[int] = set()
    replacements: list[Term] = []

    for i, lambdaTerm in enumerate(terms):
        if i in used:
            continue

        matched = False

        for tensorIndex, lambdaTensor in enumerate(lambdaTerm.tensors):
            if lambdaTensor.name != "Lambda2":
                continue

            p, q = lambdaTensor.upper
            r, s = lambdaTensor.lower
            otherTensors = (
                lambdaTerm.tensors[:tensorIndex]
                + lambdaTerm.tensors[tensorIndex + 1:]
            )
            gammaLeft = Tensor(
                name = "Gamma1",
                upper = (q,),
                lower = (s,),
            )
            gammaRight = Tensor(
                name = "Gamma1",
                upper = (p,),
                lower = (r,),
            )
            targetCoeff = lambdaTerm.coeff / 2

            for j, gammaTerm in enumerate(terms):
                if j == i or j in used:
                    continue

                if gammaTerm.coeff != targetCoeff:
                    continue

                if tuple(sorted(gammaTerm.deltas)) != tuple(sorted(lambdaTerm.deltas)):
                    continue

                baseTensors = removeTensorPair(
                    gammaTerm.tensors,
                    gammaLeft,
                    gammaRight,
                )

                if baseTensors is None:
                    continue

                if baseTensors != tuple(sorted(otherTensors)):
                    continue

                replacements.append(
                    Term(
                        coeff = lambdaTerm.coeff,
                        deltas = lambdaTerm.deltas,
                        tensors = tuple(
                            sorted(
                                otherTensors
                                + (
                                    Tensor(
                                        name = "Lambda2",
                                        upper = (q, p),
                                        lower = (s, r),
                                    ),
                                )
                            )
                        ),
                        generators = (),
                    )
                )
                used.add(i)
                used.add(j)
                matched = True
                break

            if matched:
                break

        if not matched:
            continue

    out = [
        t
        for i, t in enumerate(terms)
        if i not in used
    ] + replacements
    terms = list(combineLikeTerms(tuple(out)))
    used = set()
    replacements = []

    for i, lambdaTerm in enumerate(terms):
        if i in used:
            continue

        matched = False

        for tensorIndex, lambdaTensor in enumerate(lambdaTerm.tensors):
            if lambdaTensor.name != "Lambda2":
                continue

            p, q = lambdaTensor.upper
            s, r = lambdaTensor.lower
            otherTensors = (
                lambdaTerm.tensors[:tensorIndex]
                + lambdaTerm.tensors[tensorIndex + 1:]
            )
            gammaTensor = Tensor(
                name = "Gamma1",
                upper = (q,),
                lower = (s,),
            )
            thetaTensor = Tensor(
                name = "Theta",
                upper = (p,),
                lower = (r,),
            )

            for j, thetaTerm in enumerate(terms):
                if j == i or j in used:
                    continue

                if thetaTerm.coeff != lambdaTerm.coeff:
                    continue

                if tuple(sorted(thetaTerm.deltas)) != tuple(sorted(lambdaTerm.deltas)):
                    continue

                baseTensors = removeTensorPair(
                    thetaTerm.tensors,
                    gammaTensor,
                    thetaTensor,
                )

                if baseTensors is None:
                    continue

                if baseTensors != tuple(sorted(otherTensors)):
                    continue

                used.add(i)
                used.add(j)
                matched = True
                break

            if matched:
                break

        if not matched:
            continue

    out = [
        t
        for i, t in enumerate(terms)
        if i not in used
    ] + replacements

    return combineLikeTerms(tuple(out))


def canonicaliseSpinFreeCumulants(expr: Expr) -> Expr:
    """Apply conservative spin-free Lambda2 orientation identities."""
    current = introduceTheta(
        combineLikeTerms(expr)
    )

    while True:
        nextExpr = canonicaliseSpinFreeCumulantsOnce(current)

        if nextExpr == current:
            return current

        current = nextExpr


def introduceTheta(expr: Expr) -> Expr:
    """Rewrite simple delta/Gamma1 pairs into Theta notation."""
    terms = list(combineLikeTerms(expr))
    used: set[int] = set()
    out: list[Term] = []

    for i, deltaTerm in enumerate(terms):
        if i in used:
            continue

        if deltaTerm.generators:
            continue

        if not deltaTerm.deltas:
            continue

        matched = False

        for deltaIndex, d in enumerate(deltaTerm.deltas):
            otherDeltas = (
                deltaTerm.deltas[:deltaIndex]
                + deltaTerm.deltas[deltaIndex + 1:]
            )
            gammaPartner = Tensor(
                name = "Gamma1",
                upper = (d.right,),
                lower = (d.left,),
            )
            thetaTensor = Tensor(
                name = "Theta",
                upper = (d.right,),
                lower = (d.left,),
            )
            targetCoeff = -deltaTerm.coeff / 2

            for j, gammaTerm in enumerate(terms):
                if j == i or j in used:
                    continue

                if gammaTerm.generators:
                    continue

                if gammaTerm.coeff != targetCoeff:
                    continue

                if tuple(sorted(gammaTerm.deltas)) != tuple(sorted(otherDeltas)):
                    continue

                baseTensors = removeOneTensor(gammaTerm.tensors, gammaPartner)

                if baseTensors is None:
                    continue

                if baseTensors != tuple(sorted(deltaTerm.tensors)):
                    continue

                out.append(
                    Term(
                        coeff = deltaTerm.coeff / 2,
                        deltas = tuple(sorted(otherDeltas)),
                        tensors = tuple(sorted(baseTensors + (thetaTensor,))),
                        generators = (),
                    )
                )
                used.add(i)
                used.add(j)
                matched = True
                break

            if matched:
                break

    for i, t in enumerate(terms):
        if i not in used:
            out.append(t)

    return combineLikeTerms(tuple(out))


def finalSimplify(expr: Expr) -> Expr:
    """Apply general expression simplifications to a fixed point."""
    current = combineLikeTerms(expr)

    while True:
        nextExpr = canonicaliseTensorSymmetry(current)
        nextExpr = canonicaliseLambda3NullSums(nextExpr)
        nextExpr = canonicaliseSpinFreeCumulants(nextExpr)
        nextExpr = introduceTheta(nextExpr)
        nextExpr = combineLikeTerms(nextExpr)

        if nextExpr == current:
            return current

        current = nextExpr


def allActive(t: Tensor) -> bool:
    """Return True if all tensor indices are active."""
    return all(i.space == Space.ACTIVE for i in t.upper + t.lower)


def gamma1BySpace(t: Tensor) -> Expr:
    """Simplify Gamma1 using C/A/V orbital classes."""
    p = t.upper[0]
    q = t.lower[0]

    if p.space == Space.ACTIVE and q.space == Space.ACTIVE:
        return tensor("Gamma1", (p,), (q,))

    if p.space == Space.CORE and q.space == Space.CORE:
        return scale(delta(p, q), 2)

    return zero()


def thetaBySpace(t: Tensor) -> Expr:
    """Simplify Theta using C/A/V orbital classes."""
    p = t.upper[0]
    q = t.lower[0]

    if p.space == Space.ACTIVE and q.space == Space.ACTIVE:
        return tensor("Theta", (p,), (q,))

    if p.space == Space.VIRTUAL and q.space == Space.VIRTUAL:
        return scale(delta(p, q), 2)

    return zero()


def lambdaBySpace(t: Tensor) -> Expr:
    """Retain cumulants only when all indices are active."""
    if allActive(t):
        return tensor(t.name, t.upper, t.lower)

    return zero()


def simplifyTensorBySpace(t: Tensor) -> Expr:
    """Simplify one tensor using core/active/virtual identities."""
    if t.name == "Gamma1":
        return gamma1BySpace(t)

    if t.name == "Theta":
        return thetaBySpace(t)

    if t.name in {"Lambda2", "Lambda3", "Lambda4"}:
        return lambdaBySpace(t)

    return tensor(t.name, t.upper, t.lower)


def simplifyReferenceSpaces(expr: Expr) -> Expr:
    """Simplify tensor factors using core/active/virtual identities."""
    out: list[Term] = []

    for termIn in expr:
        expanded = term(
            coeff = termIn.coeff,
            deltas = termIn.deltas,
            generators = termIn.generators,
        )

        for tensorIn in termIn.tensors:
            expanded = mul(
                expanded,
                simplifyTensorBySpace(tensorIn),
            )

            if not expanded:
                break

        out.extend(expanded)

    return combineLikeTerms(tuple(out))


def rewriteGammaToLambda(expr: Expr) -> Expr:
    """Rewrite spin-free RDM tensors into cumulant language."""
    out: list[Term] = []

    for termIn in expr:
        expanded = term(
            coeff = termIn.coeff,
            deltas = termIn.deltas,
            generators = termIn.generators,
        )

        for tensorIn in termIn.tensors:
            expanded = mul(
                expanded,
                gammaToLambda(tensorIn),
            )

            if not expanded:
                break

        out.extend(expanded)

    reduced = simplifyReferenceSpaces(
        combineLikeTerms(tuple(out))
    )

    for termIn in reduced:
        for tensorIn in termIn.tensors:
            if tensorIn.name in {"Gamma2", "Gamma3", "Gamma4"}:
                raise NotImplementedError(
                    f"Unreduced {tensorIn.name} remained after Gamma-to-cumulant rewrite"
                )

    return finalSimplify(reduced)
