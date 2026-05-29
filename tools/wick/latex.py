# tools/wick/latex.py

from __future__ import annotations

from fractions import Fraction

from symbols import (
    Expr,
    Idx,
    Delta,
    Tensor,
    E1,
    Term,
)


def latexIdx(i: Idx) -> str:
    """Return the LaTeX label for an index."""

    return i.name


def latexIndices(indices: tuple[Idx, ...]) -> str:
    """Return concatenated LaTeX labels for a tuple of indices."""

    return "".join(latexIdx(i) for i in indices)


def latexDelta(d: Delta) -> str:
    """Return LaTeX for a Kronecker delta.

    Represents:

        delta^p_q
    """

    return rf"\delta^{{{latexIdx(d.left)}}}_{{{latexIdx(d.right)}}}"


def latexTensor(t: Tensor) -> str:
    """Return LaTeX for a scalar tensor factor."""

    upper = latexIndices(t.upper)
    lower = latexIndices(t.lower)

    if t.name == "Gamma1":
        return rf"\Gamma^{{{upper}}}_{{{lower}}}"

    if t.name == "Gamma2":
        return rf"\Gamma^{{{upper}}}_{{{lower}}}"

    if t.name == "Gamma3":
        return rf"\Gamma^{{{upper}}}_{{{lower}}}"

    if t.name == "Gamma4":
        return rf"\Gamma^{{{upper}}}_{{{lower}}}"

    if t.name == "Lambda2":
        return rf"\Lambda^{{{upper}}}_{{{lower}}}"

    if t.name == "Lambda3":
        return rf"\Lambda^{{{upper}}}_{{{lower}}}"

    if t.name == "Lambda4":
        return rf"\Lambda^{{{upper}}}_{{{lower}}}"

    if t.name == "Theta":
        return rf"\Theta^{{{upper}}}_{{{lower}}}"

    return rf"{t.name}^{{{upper}}}_{{{lower}}}"


def latexE1(e: E1) -> str:
    """Return LaTeX for an unresolved spin-free one-body generator."""

    return rf"\hat E^{{{latexIdx(e.upper)}}}_{{{latexIdx(e.lower)}}}"


def latexCoeff(c: Fraction) -> str:
    """Return LaTeX for a positive rational coefficient."""

    if c.denominator == 1:
        return str(c.numerator)

    return rf"\frac{{{c.numerator}}}{{{c.denominator}}}"


def latexTermBody(t: Term) -> str:
    """Return a LaTeX term without its leading sign."""

    coeff = abs(t.coeff)

    factors = []

    for d in t.deltas:
        factors.append(latexDelta(d))

    for tensor in t.tensors:
        factors.append(latexTensor(tensor))

    for gen in t.generators:
        factors.append(latexE1(gen))

    if coeff == 1 and factors:
        return " ".join(factors)

    if not factors:
        return latexCoeff(coeff)

    return latexCoeff(coeff) + " " + " ".join(factors)


def latexTerm(t: Term) -> str:
    """Return a signed LaTeX term."""

    body = latexTermBody(t)

    if t.coeff < 0:
        return "- " + body

    return body


def latexExpr(expr: Expr) -> str:
    """Return a LaTeX expression."""

    if not expr:
        return "0"

    pieces = []

    for i, t in enumerate(expr):
        body = latexTermBody(t)

        if i == 0:
            if t.coeff < 0:
                pieces.append("-" + body)
            else:
                pieces.append(body)
            continue

        if t.coeff < 0:
            pieces.append("- " + body)
        else:
            pieces.append("+ " + body)

    return " ".join(pieces)


def latexEquation(lhs: str, expr: Expr) -> str:
    """Return a displayed LaTeX equation."""

    return "\\begin{align}\n" + lhs + " &= " + latexExpr(expr) + "\n\\end{align}"


def printLatexExpr(expr: Expr) -> None:
    """Print a LaTeX expression."""

    print(latexExpr(expr))


def printLatexEquation(lhs: str, expr: Expr) -> None:
    """Print a displayed LaTeX equation."""

    print(latexEquation(lhs, expr))
