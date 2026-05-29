# tools/wick/symcheck.py

from __future__ import annotations

import argparse

import sympy as sp

from symbols import Expr, Term, Delta, Tensor, term, add, mul, tensor, combineLikeTerms
from classes import blockByName
from cumulants import gamma3ToLambda
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

    return sp.Symbol("__".join(str(x) for x in key), commutative = True)


def sympyTerm(t: Term) -> sp.Expr:
    """Convert one generated symbolic term into a SymPy monomial."""

    out = sp.Rational(t.coeff.numerator, t.coeff.denominator)

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


def D(p: str, q: str) -> sp.Symbol:
    return symbolForKey(("D", p, q))


def G(p: str, q: str) -> sp.Symbol:
    return symbolForKey(("G", p, q))


def T(p: str, q: str) -> sp.Symbol:
    return symbolForKey(("T", p, q))


def L2(pq: str, rs: str) -> sp.Symbol:
    p, q = pq
    r, s = rs
    return symbolForKey(min(
        ("L2", p, q, r, s),
        ("L2", q, p, s, r),
    ))


def L3(pqr: str, suv: str) -> sp.Symbol:
    return symbolForKey(("L3", pqr, suv))


def targetExpr(blockName: str) -> sp.Expr:
    """Return target expression for supported validation blocks."""

    if blockName == "C14":
        return -D("i", "j") * L2("wx", "uv")

    if blockName == "C15":
        return (
            -sp.Rational(1, 2) * G("t", "w") * L2("yz", "ux")
            -sp.Rational(1, 2) * G("t", "x") * L2("yz", "wu")
            +sp.Rational(1, 2) * L2("ty", "xw") * T("z", "u")
            +sp.Rational(1, 2) * L2("tz", "wx") * T("y", "u")
            +L3("tyz", "uwx")
        )

    if blockName == "C16":
        return (
            -sp.Rational(1, 2) * D("a", "b") * D("i", "j") * G("u", "x") * T("y", "w")
            -D("a", "b") * D("i", "j") * L2("uy", "wx")
        )

    raise ValueError(f"No target expression for {blockName}")


def rewriteOnlyGamma3(expr: Expr) -> Expr:
    """Rewrite only Gamma3 tensors, leaving other Gamma tensors unchanged."""

    out = []

    for termIn in expr:
        expanded = term(
            coeff = termIn.coeff,
            deltas = termIn.deltas,
            generators = termIn.generators,
        )

        for tensorIn in termIn.tensors:
            if tensorIn.name == "Gamma3":
                expanded = mul(
                    expanded,
                    gamma3ToLambda(tensorIn),
                )
            else:
                expanded = mul(
                    expanded,
                    tensor(tensorIn.name, tensorIn.upper, tensorIn.lower),
                )

            if not expanded:
                break

        out.extend(expanded)

    return combineLikeTerms(tuple(out))


def gamma3OnlyDiff(blockName: str) -> None:
    """Print the Gamma3-only contribution for a block."""

    from overlap import rawOverlapExpr

    block = blockByName(blockName)
    raw = rawOverlapExpr(block)
    gamma3Only = rewriteOnlyGamma3(raw)

    print(sympyExpr(gamma3Only))


def compareBlock(blockName: str) -> int:
    """Compare generated block expression against target."""

    block = blockByName(blockName)
    generated = sympyExpr(overlapExpr(block))
    target = targetExpr(blockName)
    diff = sp.expand(generated - target)

    print(f"===== {blockName} generated =====")
    print(generated)
    print(f"===== {blockName} target =====")
    print(target)
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
        choices = ["C14", "C15", "C16"],
        required = True,
    )
    parser.add_argument(
        "--gamma3-only",
        action = "store_true",
    )

    args = parser.parse_args()

    if args.gamma3_only:
        gamma3OnlyDiff(args.block)
        return

    raise SystemExit(compareBlock(args.block))


if __name__ == "__main__":
    main()
