# tools/wick/gno.py

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from fractions import Fraction
from itertools import product as cartesianProduct

from symbols import (
    Expr,
    Idx,
    Space,
    E1,
    Tensor,
    delta,
    tensor,
    term,
    zero,
    add,
    mul,
    prod,
    scale,
    combineLikeTerms,
)
from classes import SingleSpec, DoubleSpec, OverlapBlock
from cumulants import (
    ALPHA,
    BETA,
    gamma1SpinComponent,
    gammaSpinComponent,
    lambda2SpinComponent,
    permutationSign,
    setPartitions,
    simplifyReferenceSpaces,
    finalSimplify,
)


@dataclass(frozen = True)
class GnoTerm:
    """One term in a grouped generalized-normal-ordered expansion."""
    coeff: Fraction
    tensors: tuple[Tensor, ...] = ()
    deltas: tuple = ()
    groups: tuple[tuple[E1, ...], ...] = ()


@dataclass(frozen = True, order = True)
class SpinOp:
    """One spin-orbital elementary operator with a normal-order group id."""
    kind: str
    index: Idx
    spin: int
    group: int


def gnoTerm(
    coeff: int | Fraction = 1,
    *,
    tensors: tuple = (),
    deltas: tuple = (),
    groups: tuple[tuple[E1, ...], ...] = (),
) -> list[GnoTerm]:
    """Construct one grouped GNO term."""
    c = coeff if isinstance(coeff, Fraction) else Fraction(coeff)

    if c == 0:
        return []

    return [
        GnoTerm(
            coeff = c,
            tensors = tuple(tensors),
            deltas = tuple(deltas),
            groups = tuple(groups),
        )
    ]


def gnoAdd(*exprs: list[GnoTerm]) -> list[GnoTerm]:
    """Add grouped GNO expressions."""
    out = []

    for expr in exprs:
        out.extend(expr)

    return out


def gnoScale(expr: list[GnoTerm], coeff: int | Fraction) -> list[GnoTerm]:
    """Scale a grouped GNO expression."""
    c = coeff if isinstance(coeff, Fraction) else Fraction(coeff)

    if c == 0:
        return []

    return [
        GnoTerm(
            coeff = c * termIn.coeff,
            tensors = termIn.tensors,
            deltas = termIn.deltas,
            groups = termIn.groups,
        )
        for termIn in expr
        if c * termIn.coeff != 0
    ]


def gnoMul(left: list[GnoTerm], right: list[GnoTerm]) -> list[GnoTerm]:
    """Multiply grouped GNO expressions, preserving group boundaries."""
    out = []

    for a in left:
        for b in right:
            coeff = a.coeff * b.coeff

            if coeff == 0:
                continue

            out.append(
                GnoTerm(
                    coeff = coeff,
                    tensors = a.tensors + b.tensors,
                    deltas = a.deltas + b.deltas,
                    groups = a.groups + b.groups,
                )
            )

    return out


def gnoGroup(*generators: E1) -> list[GnoTerm]:
    """Construct one internally generalized-normal-ordered group."""
    return gnoTerm(
        groups = (tuple(generators),),
    )


def gnoTau1(p: Idx, q: Idx) -> list[GnoTerm]:
    """Return the grouped operator {E^p_q}.

    This is a single normal-ordered group. It is not expanded as
    E^p_q - Gamma^p_q, because the grouped Wick evaluator already knows that
    contractions internal to one normal-ordered group are forbidden.
    """
    return gnoGroup(
        E1(p, q),
    )


def gnoTau2(p: Idx, q: Idx, r: Idx, s: Idx) -> list[GnoTerm]:
    """Return the grouped operator {E^{pq}_{rs}}.

    The group stores the spin-orbital string

        a†_p a†_q a_s a_r

    via two spin-free E1 generators. This is the operator-level GNO
    representation corresponding to the spin-free generators in Eqs. 12--15 of
    the GNOCC paper, not the scalar expansion of Eq. 15.
    """
    return gnoGroup(
        E1(p, r),
        E1(q, s),
    )


def gnoExcitation(spec: SingleSpec | DoubleSpec) -> list[GnoTerm]:
    """Return the grouped GNO expansion for one excitation spec."""
    if isinstance(spec, SingleSpec):
        return gnoTau1(
            spec.create,
            spec.annihilate,
        )

    if isinstance(spec, DoubleSpec):
        return gnoTau2(
            spec.create1,
            spec.create2,
            spec.annihilate1,
            spec.annihilate2,
        )

    raise TypeError(f"unsupported excitation spec {type(spec)}")


def gnoAdjointExcitation(spec: SingleSpec | DoubleSpec) -> list[GnoTerm]:
    """Return the grouped GNO expansion for one adjoint excitation spec."""
    if isinstance(spec, SingleSpec):
        return gnoTau1(
            spec.annihilate,
            spec.create,
        )

    if isinstance(spec, DoubleSpec):
        return gnoTau2(
            spec.annihilate1,
            spec.annihilate2,
            spec.create1,
            spec.create2,
        )

    raise TypeError(f"unsupported excitation spec {type(spec)}")


def spinOpsFromGroup(group: tuple[E1, ...], groupId: int) -> list[tuple[SpinOp, ...]]:
    """Expand one spin-free normal-order group into spin-orbital strings."""
    if len(group) == 1:
        g = group[0]

        return [
            (
                SpinOp("create", g.upper, spin, groupId),
                SpinOp("annihilate", g.lower, spin, groupId),
            )
            for spin in (ALPHA, BETA)
        ]

    if len(group) == 2:
        g1, g2 = group
        out = []

        for spin1, spin2 in cartesianProduct((ALPHA, BETA), repeat = 2):
            out.append(
                (
                    SpinOp("create", g1.upper, spin1, groupId),
                    SpinOp("create", g2.upper, spin2, groupId),
                    SpinOp("annihilate", g2.lower, spin2, groupId),
                    SpinOp("annihilate", g1.lower, spin1, groupId),
                )
            )

        return out

    raise NotImplementedError("normal-order groups above rank two are not implemented")


def spinOpsFromGroups(groups: tuple[tuple[E1, ...], ...]) -> list[tuple[SpinOp, ...]]:
    """Expand grouped spin-free operators into spin-orbital strings."""
    expansions = [()]

    for groupId, group in enumerate(groups):
        nextExpansions = []

        for prefix in expansions:
            for suffix in spinOpsFromGroup(group, groupId):
                nextExpansions.append(prefix + suffix)

        expansions = nextExpansions

    return expansions


def blockIsInternal(ops: tuple[SpinOp, ...], block: tuple[int, ...]) -> bool:
    """Return True if a generalized Wick block is internal to one GNO group."""
    groupIds = {
        ops[i].group
        for i in block
    }

    return len(groupIds) == 1


def frozenExpectationContraction(left: SpinOp, right: SpinOp) -> Expr:
    """Return one deterministic core/virtual two-point expectation."""
    if left.spin != right.spin:
        return zero()

    if left.index.space != right.index.space:
        return zero()

    if left.index.space == Space.CORE:
        if left.kind == "create" and right.kind == "annihilate":
            return delta(left.index, right.index)

        return zero()

    if left.index.space == Space.VIRTUAL:
        if left.kind == "annihilate" and right.kind == "create":
            return delta(left.index, right.index)

        return zero()

    return zero()


def activeExpectationContraction(left: SpinOp, right: SpinOp) -> Expr:
    """Return the active anticommutator contraction a_p a†_q."""
    if left.kind != "annihilate" or right.kind != "create":
        return zero()

    if left.spin != right.spin:
        return zero()

    if left.index.space != Space.ACTIVE or right.index.space != Space.ACTIVE:
        return zero()

    return delta(left.index, right.index)


def gamma2SpinComponent(
    upper: tuple,
    lower: tuple,
    upperSpins: tuple[int, int],
    lowerSpins: tuple[int, int],
) -> Expr:
    """Return one spin-orbital Gamma2 component in spin-free cumulants."""
    p, q = upper
    r, s = lower

    return add(
        mul(
            gamma1SpinComponent(
                p,
                r,
                upperSpins[0],
                lowerSpins[0],
            ),
            gamma1SpinComponent(
                q,
                s,
                upperSpins[1],
                lowerSpins[1],
            ),
        ),
        scale(
            mul(
                gamma1SpinComponent(
                    p,
                    s,
                    upperSpins[0],
                    lowerSpins[1],
                ),
                gamma1SpinComponent(
                    q,
                    r,
                    upperSpins[1],
                    lowerSpins[0],
                ),
            ),
            -1,
        ),
        lambda2SpinComponent(
            upper,
            lower,
            upperSpins,
            lowerSpins,
        ),
    )


def activeNormalRdmValue(ops: tuple[SpinOp, ...]) -> Expr:
    """Return the active RDM of a normal-ordered active string."""
    creators = tuple(op for op in ops if op.kind == "create")
    annihilators = tuple(op for op in ops if op.kind == "annihilate")

    if len(creators) != len(annihilators):
        return zero()

    upper = tuple(op.index for op in creators)
    upperSpins = tuple(op.spin for op in creators)

    lower = tuple(op.index for op in reversed(annihilators))
    lowerSpins = tuple(op.spin for op in reversed(annihilators))

    if not upper:
        return term()

    return gammaSpinComponent(
        upper,
        lower,
        upperSpins,
        lowerSpins,
    )


@cache
def activeExpectationValue(ops: tuple[SpinOp, ...]) -> Expr:
    """Evaluate an active spin-orbital operator string exactly."""
    if not ops:
        return term()

    if any(op.index.space != Space.ACTIVE for op in ops):
        raise ValueError("non-active operator reached active expectation")

    nCreate = sum(1 for op in ops if op.kind == "create")
    nAnnihilate = sum(1 for op in ops if op.kind == "annihilate")

    if nCreate != nAnnihilate:
        return zero()

    for i in range(len(ops) - 1):
        left = ops[i]
        right = ops[i + 1]

        if left.kind != "annihilate" or right.kind != "create":
            continue

        contraction = activeExpectationContraction(
            left,
            right,
        )
        noPair = ops[:i] + ops[i + 2:]
        swapped = ops[:i] + (right, left) + ops[i + 2:]

        return combineLikeTerms(
            add(
                mul(
                    contraction,
                    activeExpectationValue(noPair),
                ),
                scale(
                    activeExpectationValue(swapped),
                    -1,
                ),
            )
        )

    return activeNormalRdmValue(ops)


def firstFrozenPosition(ops: tuple[SpinOp, ...]) -> int | None:
    """Return the first core/virtual operator position."""
    for i, op in enumerate(ops):
        if op.index.space != Space.ACTIVE:
            return i

    return None


@cache
def exactExpectationValue(ops: tuple[SpinOp, ...]) -> Expr:
    """Evaluate an arbitrary spin-orbital block expectation exactly."""
    if not ops:
        return term()

    nCreate = sum(1 for op in ops if op.kind == "create")
    nAnnihilate = sum(1 for op in ops if op.kind == "annihilate")

    if nCreate != nAnnihilate:
        return zero()

    frozenPos = firstFrozenPosition(ops)

    if frozenPos is None:
        return activeExpectationValue(ops)

    out = zero()

    for j in range(frozenPos + 1, len(ops)):
        contraction = frozenExpectationContraction(
            ops[frozenPos],
            ops[j],
        )

        if not contraction:
            continue

        rest = (
            ops[:frozenPos]
            + ops[frozenPos + 1:j]
            + ops[j + 1:]
        )
        sign = -1 if (j - frozenPos - 1) % 2 else 1

        out = add(
            out,
            scale(
                mul(
                    contraction,
                    exactExpectationValue(rest),
                ),
                sign,
            ),
        )

    return combineLikeTerms(out)


def partitionSign(partition: tuple[tuple[int, ...], ...]) -> int:
    """Return the sign for collecting operators into partition blocks."""
    sequence = tuple(
        i
        for block in partition
        for i in block
    )

    return permutationSign(sequence)


@cache
def connectedCumulantValue(ops: tuple[SpinOp, ...]) -> Expr:
    """Return the connected cumulant of an ordered operator block."""
    if not ops:
        return zero()

    nCreate = sum(1 for op in ops if op.kind == "create")
    nAnnihilate = sum(1 for op in ops if op.kind == "annihilate")

    if nCreate != nAnnihilate:
        return zero()

    out = exactExpectationValue(ops)
    positions = tuple(range(len(ops)))

    for partition in setPartitions(positions):
        if len(partition) == 1:
            continue

        factors = []
        valid = True

        for block in partition:
            blockOps = tuple(ops[i] for i in block)
            value = connectedCumulantValue(blockOps)

            if not value:
                valid = False
                break

            factors.append(value)

        if not valid:
            continue

        out = add(
            out,
            scale(
                prod(factors),
                -partitionSign(partition),
            ),
        )

    return combineLikeTerms(out)


def evaluateSpinString(ops: tuple[SpinOp, ...]) -> Expr:
    """Evaluate one product of GNO groups by generalized Wick's theorem."""
    out = zero()
    positions = tuple(range(len(ops)))

    for partition in setPartitions(positions):
        factors = []
        valid = True

        for block in partition:
            if blockIsInternal(ops, block):
                valid = False
                break

            blockOps = tuple(ops[i] for i in block)
            value = connectedCumulantValue(blockOps)

            if not value:
                valid = False
                break

            factors.append(value)

        if not valid:
            continue

        out = add(
            out,
            scale(
                prod(factors),
                partitionSign(partition),
            ),
        )

    return combineLikeTerms(out)


def evaluateGroupedTerm(termIn: GnoTerm) -> Expr:
    """Evaluate one scalar factor times grouped GNO operator product."""
    scalar = term(
        coeff = termIn.coeff,
        deltas = termIn.deltas,
        tensors = termIn.tensors,
    )

    if not termIn.groups:
        return scalar

    out = zero()

    for ops in spinOpsFromGroups(termIn.groups):
        out = add(
            out,
            evaluateSpinString(ops),
        )

    return mul(
        scalar,
        combineLikeTerms(out),
    )


def gnoExpectation(expr: list[GnoTerm]) -> Expr:
    """Evaluate a grouped GNO expression."""
    out = zero()

    for termIn in expr:
        out = add(
            out,
            evaluateGroupedTerm(termIn),
        )

    return finalSimplify(
        simplifyReferenceSpaces(
            combineLikeTerms(out)
        )
    )


def gnoOverlapBlock(block: OverlapBlock) -> Expr:
    """Evaluate one overlap block by generic grouped GNO Wick algebra."""
    return gnoExpectation(
        gnoMul(
            gnoAdjointExcitation(block.left),
            gnoExcitation(block.right),
        )
    )
