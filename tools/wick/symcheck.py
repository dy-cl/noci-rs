# tools/wick/symcheck.py

from __future__ import annotations

import argparse
from fractions import Fraction

import sympy as sp

from symbols import (
    Expr,
    Term,
    Delta,
    Tensor,
    delta,
    tensor,
    add,
    mul,
    scale,
)

from classes import (
    blockByName,
    availableBlocks,
    core,
    active,
    virtual,
)
from latex import latexEquation
from overlap import overlapExpr


def idxKey(indices) -> str:
    """Return a stable key for a tuple of symbolic indices."""
    return "".join(i.name for i in indices)


def deltaKey(d: Delta) -> tuple:
    """Return a canonical delta key."""
    return (
        "D",
        d.left.name,
        d.right.name,
    )


def lambda2CanonicalKey(t: Tensor) -> tuple:
    """Return canonical Lambda2 key using pair-exchange symmetry.

    Uses:
        Lambda^{pq}_{rs} = Lambda^{qp}_{sr}
    """
    p, q = t.upper
    r, s = t.lower

    direct = (
        "L2",
        p.name,
        q.name,
        r.name,
        s.name,
    )
    swapped = (
        "L2",
        q.name,
        p.name,
        s.name,
        r.name,
    )

    return min(direct, swapped)


def tensorKey(t: Tensor) -> tuple:
    """Return a canonical tensor key for SymPy comparison."""
    if t.name == "Gamma1":
        return (
            "G",
            idxKey(t.upper),
            idxKey(t.lower),
        )

    if t.name == "Theta":
        return (
            "T",
            idxKey(t.upper),
            idxKey(t.lower),
        )

    if t.name == "Lambda2":
        return lambda2CanonicalKey(t)

    if t.name == "Lambda3":
        return (
            "L3",
            idxKey(t.upper),
            idxKey(t.lower),
        )

    if t.name == "Lambda4":
        return (
            "L4",
            idxKey(t.upper),
            idxKey(t.lower),
        )

    if t.name in {"Gamma2", "Gamma3", "Gamma4"}:
        raise ValueError(f"Unreduced tensor in final expression: {t}")

    return (
        t.name,
        idxKey(t.upper),
        idxKey(t.lower),
    )


def symbolForKey(key: tuple) -> sp.Symbol:
    """Return a commutative SymPy symbol for a tensor or delta key."""
    return sp.Symbol(
        "__".join(str(x) for x in key),
        commutative = True,
    )


def sympyTerm(t: Term) -> sp.Expr:
    """Convert one generated symbolic term into a SymPy monomial."""
    out = sp.Rational(
        t.coeff.numerator,
        t.coeff.denominator,
    )

    for d in t.deltas:
        out *= symbolForKey(deltaKey(d))

    for tensorIn in t.tensors:
        out *= symbolForKey(tensorKey(tensorIn))

    if t.generators:
        raise ValueError("Cannot compare unresolved generator terms.")

    return out


def sympyExpr(expr: Expr) -> sp.Expr:
    """Convert a generated expression into a canonical SymPy expression."""
    out = sp.Integer(0)

    for termIn in expr:
        out += sympyTerm(termIn)

    return sp.expand(out)


def idx(name: str):
    """Return an index object using the paper's C/A/V naming convention."""
    if name in {"i", "j", "k", "l"}:
        return core(name)

    if name in {"a", "b", "c", "d"}:
        return virtual(name)

    return active(name)


def Dn(p: str, q: str) -> Expr:
    """Return a native delta expression."""
    return delta(
        idx(p),
        idx(q),
    )


def Gn(p: str, q: str) -> Expr:
    """Return a native Gamma1 expression."""
    return tensor(
        "Gamma1",
        (idx(p),),
        (idx(q),),
    )


def Tn(p: str, q: str) -> Expr:
    """Return a native Theta expression."""
    return tensor(
        "Theta",
        (idx(p),),
        (idx(q),),
    )


def L2n(pq: str, rs: str) -> Expr:
    """Return a native Lambda2 expression."""
    return tensor(
        "Lambda2",
        tuple(idx(p) for p in pq),
        tuple(idx(r) for r in rs),
    )


def L3n(pqr: str, suv: str) -> Expr:
    """Return a native Lambda3 expression."""
    return tensor(
        "Lambda3",
        tuple(idx(p) for p in pqr),
        tuple(idx(s) for s in suv),
    )


def targetNativeExpr(blockName: str) -> Expr:
    """Return target expression as the generator's native symbolic Expr."""
    if blockName == "C1":
        return mul(
            Dn("i", "j"),
            Tn("v", "u"),
        )

    if blockName == "C2":
        return mul(
            Dn("a", "b"),
            Gn("t", "u"),
        )

    if blockName == "C3":
        return add(
            scale(
                mul(
                    Gn("u", "w"),
                    Tn("x", "v"),
                ),
                Fraction(1, 2),
            ),
            L2n("ux", "vw"),
        )

    if blockName == "C4":
        return mul(
            mul(
                Dn("i", "j"),
                Dn("a", "b"),
            ),
            add(
                mul(
                    Gn("u", "w"),
                    Tn("x", "v"),
                ),
                scale(
                    L2n("ux", "wv"),
                    -1,
                ),
            ),
        )

        
    if blockName == "C5":
        return mul(
            mul(
                Dn("i", "j"),
                Dn("a", "b"),
            ),
            add(
                mul(
                    Gn("u", "w"),
                    Tn("x", "v"),
                ),
                scale(
                    L2n("ux", "vw"),
                    2,
                ),
            ),
        )


    if blockName == "C6":
        return mul(
            mul(
                Dn("i", "j"),
                Gn("u", "v"),
            ),
            add(
                scale(
                    mul(
                        Dn("a", "c"),
                        Dn("b", "d"),
                    ),
                    2,
                ),
                scale(
                    mul(
                        Dn("a", "d"),
                        Dn("b", "c"),
                    ),
                    -1,
                ),
            ),
        )

    if blockName == "C7":
        return mul(
            mul(
                Dn("a", "b"),
                Tn("v", "u"),
            ),
            add(
                scale(
                    mul(
                        Dn("i", "k"),
                        Dn("j", "l"),
                    ),
                    2,
                ),
                scale(
                    mul(
                        Dn("i", "l"),
                        Dn("j", "k"),
                    ),
                    -1,
                ),
            ),
        )

    if blockName == "C8":
        return add(
            mul(
                mul(
                    Dn("i", "k"),
                    Dn("j", "l"),
                ),
                add(
                    mul(
                        Tn("w", "u"),
                        Tn("x", "v"),
                    ),
                    scale(
                        mul(
                            Tn("w", "v"),
                            Tn("x", "u"),
                        ),
                        Fraction(-1, 2),
                    ),
                    L2n("wx", "uv"),
                ),
            ),
            mul(
                mul(
                    Dn("i", "l"),
                    Dn("j", "k"),
                ),
                add(
                    mul(
                        Tn("w", "v"),
                        Tn("x", "u"),
                    ),
                    scale(
                        mul(
                            Tn("w", "u"),
                            Tn("x", "v"),
                        ),
                        Fraction(-1, 2),
                    ),
                    L2n("wx", "vu"),
                ),
            ),
        )

    if blockName == "C9":
        return mul(
            Dn("i", "j"),
            add(
                scale(
                    L3n("uyz", "wvx"),
                    -1,
                ),
                scale(
                    mul(
                        Tn("y", "w"),
                        L2n("uz", "vx"),
                    ),
                    Fraction(-1, 2),
                ),
                scale(
                    mul(
                        Tn("z", "w"),
                        L2n("uy", "xv"),
                    ),
                    Fraction(-1, 2),
                ),
                scale(
                    mul(
                        Tn("z", "v"),
                        L2n("uy", "wx"),
                    ),
                    Fraction(-1, 2),
                ),
                mul(
                    Tn("y", "v"),
                    L2n("uz", "wx"),
                ),
                scale(
                    mul(
                        Gn("u", "x"),
                        add(
                            mul(
                                Tn("z", "w"),
                                Tn("y", "v"),
                            ),
                            scale(
                                mul(
                                    Tn("z", "v"),
                                    Tn("y", "w"),
                                ),
                                Fraction(-1, 2),
                            ),
                            L2n("yz", "vw"),
                        ),
                    ),
                    Fraction(1, 2),
                ),
            ),
        )

    if blockName == "C10":
        return mul(
            Dn("a", "b"),
            add(
                scale(
                    mul(
                        Tn("z", "v"),
                        add(
                            mul(
                                Gn("t", "x"),
                                Gn("u", "y"),
                            ),
                            scale(
                                mul(
                                    Gn("t", "y"),
                                    Gn("u", "x"),
                                ),
                                Fraction(-1, 2),
                            ),
                            L2n("tu", "xy"),
                        ),
                    ),
                    Fraction(1, 2),
                ),
                scale(
                    mul(
                        Gn("t", "x"),
                        L2n("uz", "yv"),
                    ),
                    Fraction(-1, 2),
                ),
                scale(
                    mul(
                        Gn("t", "y"),
                        L2n("uz", "vx"),
                    ),
                    Fraction(-1, 2),
                ),
                mul(
                    Gn("u", "y"),
                    L2n("tz", "vx"),
                ),
                scale(
                    mul(
                        Gn("u", "x"),
                        L2n("tz", "vy"),
                    ),
                    Fraction(-1, 2),
                ),
                L3n("tuz", "vyx"),
            ),
        )

    if blockName == "C11":
        return add(
            mul(
                mul(
                    Dn("a", "c"),
                    Dn("b", "d"),
                ),
                add(
                    mul(
                        Gn("t", "v"),
                        Gn("u", "w"),
                    ),
                    scale(
                        mul(
                            Gn("t", "w"),
                            Gn("u", "v"),
                        ),
                        Fraction(-1, 2),
                    ),
                    L2n("tu", "vw"),
                ),
            ),
            mul(
                mul(
                    Dn("a", "d"),
                    Dn("b", "c"),
                ),
                add(
                    mul(
                        Gn("u", "v"),
                        Gn("t", "w"),
                    ),
                    scale(
                        mul(
                            Gn("u", "w"),
                            Gn("t", "v"),
                        ),
                        Fraction(-1, 2),
                    ),
                    L2n("ut", "vw"),
                ),
            ),
        )

    if blockName == "C13":
        return mul(
            Dn("a", "b"),
            L2n("ux", "wv"),
        )

    if blockName == "C14":
        return scale(
            mul(
                Dn("i", "j"),
                L2n("wx", "uv"),
            ),
            -1,
        )

    if blockName == "C15":
        return add(
            scale(
                mul(
                    Gn("t", "w"),
                    L2n("yz", "ux"),
                ),
                Fraction(-1, 2),
            ),
            scale(
                mul(
                    Gn("t", "x"),
                    L2n("yz", "wu"),
                ),
                Fraction(-1, 2),
            ),
            scale(
                mul(
                    L2n("ty", "xw"),
                    Tn("z", "u"),
                ),
                Fraction(1, 2),
            ),
            scale(
                mul(
                    L2n("tz", "wx"),
                    Tn("y", "u"),
                ),
                Fraction(1, 2),
            ),
            L3n("tyz", "uwx"),
        )

    if blockName == "C16":
        return add(
            scale(
                mul(
                    mul(
                        Dn("a", "b"),
                        Dn("i", "j"),
                    ),
                    mul(
                        Gn("u", "x"),
                        Tn("y", "w"),
                    ),
                ),
                Fraction(-1, 2),
            ),
            scale(
                mul(
                    mul(
                        Dn("a", "b"),
                        Dn("i", "j"),
                    ),
                    L2n("uy", "wx"),
                ),
                -1,
            ),
        )

    raise NotImplementedError(f"no target expression for {blockName}")

def targetBlockNames() -> list[str]:
    """Return blocks with trusted hand-transcribed target expressions."""
    return [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C13",
        "C14",
        "C15",
        "C16",
    ]

def missingTargetBlockNames() -> list[str]:
    """Return implemented blocks needing supplementary target expressions."""
    return []

def expectedUnsupportedBlocks() -> dict[str, str]:
    """Return blocks that should fail cleanly for known theoretical reasons."""
    return {
        "C12": "All-active spin-free Gamma4/C12 is unsupported",
    }

def assertNoUnreducedGamma(expr: Expr) -> None:
    """Raise if a reduced expression still contains high-rank Gamma tensors."""
    for termIn in expr:
        for tensorIn in termIn.tensors:
            if tensorIn.name in {"Gamma2", "Gamma3", "Gamma4"}:
                raise ValueError(
                    f"Unreduced {tensorIn.name} remained in final expression: {tensorIn}"
                )

def smokeBlock(blockName: str) -> int:
    """Check that a block emits a reduced expression without target comparison."""
    block = blockByName(blockName)

    try:
        generatedNative = overlapExpr(block)
    except NotImplementedError as err:
        expected = expectedUnsupportedBlocks()

        if blockName in expected and expected[blockName] in str(err):
            print(f"===== {blockName} expected unsupported =====")
            print(err)
            print(f"{blockName}: EXPECTED-UNSUPPORTED")
            return 0

        print(f"===== {blockName} unexpected failure =====")
        print(err)
        print(f"{blockName}: FAIL")
        return 1

    assertNoUnreducedGamma(generatedNative)

    print(f"===== {blockName} generated latex =====")
    print(latexEquation(blockName, generatedNative))

    if blockName in missingTargetBlockNames():
        print(f"{blockName}: SMOKE-OK, NO TRUSTED TARGET YET")
        return 0

    if blockName in expectedUnsupportedBlocks():
        print(f"{blockName}: FAIL")
        print("Expected this block to fail cleanly, but it generated an expression.")
        return 1

    print(f"{blockName}: SMOKE-OK")
    return 0


def compareOrSmokeBlock(blockName: str) -> int:
    """Run exact target comparison if available, otherwise smoke-check."""
    if blockName in targetBlockNames():
        return compareBlock(blockName)

    return smokeBlock(blockName)


def checkAllBlocks(strict: bool) -> int:
    """Check every registered block.

    strict = True:
        require every non-unsupported block to have a trusted target.

    strict = False:
        exact-check trusted targets; smoke-check targetless blocks.
    """
    status = 0

    for blockName in availableBlocks():
        print(f"\n######## {blockName} ########")

        if strict and blockName in missingTargetBlockNames():
            print(f"{blockName}: FAIL")
            print("No trusted target expression has been transcribed yet.")
            status = 1
            continue

        result = compareOrSmokeBlock(blockName)

        if result != 0:
            status = result

    return status

def compareBlock(blockName: str) -> int:
    """Compare generated block expression against target."""
    block = blockByName(blockName)
    generatedNative = overlapExpr(block)
    targetNative = targetNativeExpr(blockName)

    generated = sympyExpr(generatedNative)
    target = sympyExpr(targetNative)
    diff = sp.expand(generated - target)

    print(f"===== {blockName} generated latex =====")
    print(latexEquation(blockName, generatedNative))
    print(f"===== {blockName} target latex =====")
    print(latexEquation(f"{blockName}_target", targetNative))
    print(f"===== {blockName} diff =====")
    print(diff)

    if diff == 0:
        print(f"{blockName}: OK")
        return 0

    print(f"{blockName}: FAIL")
    return 1

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices = availableBlocks(),
    )
    parser.add_argument(
        "--all",
        action = "store_true",
        help = "Check all registered blocks.",
    )
    parser.add_argument(
        "--strict",
        action = "store_true",
        help = "Fail if any implemented block lacks a trusted target.",
    )

    args = parser.parse_args()

    if args.all:
        raise SystemExit(checkAllBlocks(args.strict))

    if args.block is None:
        parser.error("one of --block or --all is required")

    raise SystemExit(compareOrSmokeBlock(args.block))

if __name__ == "__main__":
    main()
