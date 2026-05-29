# tools/wick/cumulants.py

from __future__ import annotations
from itertools import permutations, product as cartesianProduct
from fractions import Fraction

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
    """Return one spin-orbital cumulant block in spin-free form."""

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
        permutationSign(upperSeq) * permutationSign(lowerSeq)
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

    This is a local Gamma3 reconstruction term. It must not inspect overlap
    block names or final expression shape.
    """

    if not allActive(t):
        return zero()

    p, q, r = t.upper
    s, u, v = t.lower

    return add(
        tensor("Lambda3", (p, q, r), (s, u, v)),
        scale(
            mul(
                tensor("Gamma1", (p,), (u,)),
                tensor("Lambda2", (q, r), (s, v)),
            ),
            Fraction(-1, 2),
        ),
        scale(
            mul(
                tensor("Gamma1", (p,), (v,)),
                tensor("Lambda2", (q, r), (u, s)),
            ),
            Fraction(-1, 2),
        ),
        scale(
            mul(
                tensor("Theta", (q,), (s,)),
                tensor("Lambda2", (p, r), (u, v)),
            ),
            Fraction(1, 2),
        ),
        scale(
            mul(
                tensor("Theta", (r,), (s,)),
                tensor("Lambda2", (p, q), (v, u)),
            ),
            Fraction(1, 2),
        ),
        scale(
            mul(
                tensor("Gamma1", (p,), (s,)),
                tensor("Lambda2", (q, r), (v, u)),
            ),
            Fraction(1, 2),
        ),
        scale(
            mul(
                tensor("Gamma1", (p,), (u,)),
                tensor("Lambda2", (q, r), (s, v)),
            ),
            Fraction(1, 2),
        ),
        scale(
            mul(
                tensor("Gamma1", (p,), (u,)),
                tensor("Lambda2", (q, r), (v, s)),
            ),
            Fraction(-1, 4),
        ),
        scale(
            mul(
                mul(
                    tensor("Gamma1", (p,), (v,)),
                    tensor("Gamma1", (q,), (u,)),
                ),
                tensor("Theta", (r,), (s,)),
            ),
            Fraction(-1, 4),
        ),
        scale(
            mul(
                tensor("Gamma1", (p,), (v,)),
                tensor("Lambda2", (q, r), (s, u)),
            ),
            Fraction(-1, 4),
        ),
        scale(
            mul(
                tensor("Gamma1", (p,), (v,)),
                tensor("Lambda2", (q, r), (u, s)),
            ),
            Fraction(1, 2),
        ),
        scale(
            mul(
                tensor("Gamma1", (q,), (s,)),
                tensor("Lambda2", (p, r), (v, u)),
            ),
            Fraction(-1, 4),
        ),
        scale(
            mul(
                tensor("Gamma1", (q,), (u,)),
                tensor("Lambda2", (p, r), (s, v)),
            ),
            Fraction(-1, 2),
        ),
        scale(
            mul(
                tensor("Gamma1", (q,), (u,)),
                tensor("Lambda2", (p, r), (v, s)),
            ),
            Fraction(1, 2),
        ),
        scale(
            mul(
                tensor("Gamma1", (q,), (v,)),
                tensor("Lambda2", (p, r), (u, s)),
            ),
            Fraction(-1, 4),
        ),
        scale(
            mul(
                tensor("Gamma1", (r,), (s,)),
                tensor("Lambda2", (p, q), (u, v)),
            ),
            Fraction(-1, 4),
        ),
        scale(
            mul(
                tensor("Gamma1", (r,), (u,)),
                tensor("Lambda2", (p, q), (v, s)),
            ),
            Fraction(-1, 4),
        ),
        scale(
            mul(
                tensor("Gamma1", (r,), (v,)),
                tensor("Lambda2", (p, q), (s, u)),
            ),
            Fraction(-1, 2),
        ),
        scale(
            mul(
                tensor("Gamma1", (r,), (v,)),
                tensor("Lambda2", (p, q), (u, s)),
            ),
            Fraction(1, 2),
        ),
        scale(
            mul(
                tensor("Lambda2", (p, q), (v, u)),
                tensor("Theta", (r,), (s,)),
            ),
            Fraction(-1, 2),
        ),
        scale(
            mul(
                tensor("Lambda2", (p, r), (u, v)),
                tensor("Theta", (q,), (s,)),
            ),
            Fraction(-1, 2),
        ),
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


def gamma4MixedConnectedContribution(t: Tensor) -> Expr:
    """Return mixed core/virtual rank-four connected reference contractions."""

    p, q, r, s = t.upper
    u, v, w, x = t.lower

    if (
        p.space == Space.CORE
        and q.space == Space.ACTIVE
        and r.space == Space.VIRTUAL
        and s.space == Space.ACTIVE
        and u.space == Space.ACTIVE
        and v.space == Space.VIRTUAL
        and w.space == Space.CORE
        and x.space == Space.ACTIVE
    ):
        return scale(
            mul(
                mul(
                    delta(v, r),
                    delta(p, w),
                ),
                tensor("Lambda2", (q, s), (x, u)),
            ),
            Fraction(-1, 2),
        )

    return zero()


def gamma4BySpinSummation(t: Tensor) -> Expr:
    """Rewrite Gamma4 by spin-summing spin-orbital cumulant products."""

    positions = (0, 1, 2, 3)
    out = add(
        tensor("Lambda4", t.upper, t.lower),
        gamma4MixedConnectedContribution(t),
    )

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
    """Return all set partitions of an index-position tuple.

    Example:
        (0, 1, 2) gives partitions such as:

            ((0, 1, 2),)
            ((0,), (1, 2))
            ((0,), (1,), (2,))
    """

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
    """Rewrite simple delta/Gamma1 pairs into Theta notation.

    Uses:

        Theta^x_v = 2 delta^v_x - Gamma^x_v

    Therefore:

        c delta^v_x A - c/2 A Gamma^x_v
        =
        c/2 A Theta^x_v

    where A is any product of scalar tensors.
    """

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
    """Simplify Gamma1 using C/A/V orbital classes.

    Core:
        Gamma^i_j = 2 delta^i_j

    Active:
        Gamma^t_u is retained.

    Virtual:
        Gamma^a_b = 0

    Mixed-space Gamma1 blocks are zero.
    """

    p = t.upper[0]
    q = t.lower[0]

    if p.space == Space.ACTIVE and q.space == Space.ACTIVE:
        return tensor("Gamma1", (p,), (q,))

    if p.space == Space.CORE and q.space == Space.CORE:
        return scale(delta(p, q), 2)

    return zero()


def thetaBySpace(t: Tensor) -> Expr:
    """Simplify Theta using C/A/V orbital classes.

    Theta^p_q = 2 delta^p_q - Gamma^p_q

    Core:
        Theta^i_j = 0

    Active:
        Theta^t_u is retained.

    Virtual:
        Theta^a_b = 2 delta^a_b
    """

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
    """Simplify one tensor using C/A/V reference-space structure."""

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

    return finalSimplify(
        simplifyReferenceSpaces(
            combineLikeTerms(tuple(out))
        )
    )
