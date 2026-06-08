from __future__ import annotations

from fractions import Fraction

from core import Delta, Expr, Idx, Tensor, Term
from specs import EXCITATIONS, overlapBlock

def latexIndex(idx: Idx) -> str:
    """Render one orbital index.

    Notation:
        i, u, a
    """
    return idx.name

def latexIndexTuple(indices: tuple[Idx, ...]) -> str:
    """Render an upper or lower tensor index tuple.

    Notation:
        (u, x, v) -> uxv
    """
    return "".join(
        latexIndex(idx)
        for idx in indices
    )

def latexDelta(delta: Delta) -> str:
    """Render a Kronecker delta.

    Notation:
        delta^i_j
    """
    return (
        "\\delta"
        + f"^{{{latexIndex(delta.left)}}}"
        + f"_{{{latexIndex(delta.right)}}}"
    )

def latexTensor(tensor: Tensor) -> str:
    """Render a spin-free tensor.

    Notation:
        Gamma1  -> Gamma
        Theta   -> Theta
        Lambda2 -> Lambda
        Lambda3 -> Lambda
        Lambda4 -> Lambda
    """
    if tensor.name == "Gamma1":
        symbol = "\\Gamma"
    elif tensor.name == "Theta":
        symbol = "\\Theta"
    elif tensor.name in {"Lambda2", "Lambda3", "Lambda4"}:
        symbol = "\\Lambda"
    else:
        symbol = tensor.name

    return (
        symbol
        + f"^{{{latexIndexTuple(tensor.upper)}}}"
        + f"_{{{latexIndexTuple(tensor.lower)}}}"
    )

def latexCoeff(coeff: Fraction, hasFactors: bool) -> str:
    """Render an absolute coefficient.

    Notation:
        1      -> omitted if there are tensor/delta factors
        1/2    -> \\frac{1}{2}
        3      -> 3
    """
    c = abs(coeff)

    if c == 1 and hasFactors:
        return ""

    if c.denominator == 1:
        return str(c.numerator)

    return f"\\frac{{{c.numerator}}}{{{c.denominator}}}"

def latexTermBody(term: Term) -> str:
    """Render one term without leading sign.

    Notation:
        c delta Gamma Lambda
    """
    factors = [
        latexDelta(delta)
        for delta in term.deltas
    ] + [
        latexTensor(tensor)
        for tensor in term.tensors
    ]

    coeff = latexCoeff(
        term.coeff,
        bool(factors),
    )

    if coeff and factors:
        return coeff + " " + " ".join(factors)

    if coeff:
        return coeff

    return " ".join(factors)

def latexTermSigned(term: Term, first: bool) -> str:
    """Render one signed term.

    Notation:
        first positive term has no leading +
    """
    body = latexTermBody(term)

    if first:
        if term.coeff < 0:
            return "-" + body

        return body

    if term.coeff < 0:
        return " - " + body

    return " + " + body

def sortedTerms(expr: Expr) -> tuple[Term, ...]:
    """Stable display ordering.

    Notation:
        Sort by deltas, tensors, then coefficient.
    """
    return tuple(sorted(
        expr,
        key = lambda term: (
            tuple(str(delta) for delta in term.deltas),
            tuple(str(tensor) for tensor in term.tensors),
            term.coeff,
        ),
    ))

def latexExpr(expr: Expr) -> str:
    """Render a complete expression on one line.

    Notation:
        term_1 + term_2 + ...
    """
    if not expr:
        return "0"

    return "".join(
        latexTermSigned(
            term,
            i == 0,
        )
        for i, term in enumerate(sortedTerms(expr))
    )

def latexExprMultiline(name: str, expr: Expr, lineWidth: int) -> str:
    """Render an expression over multiple split lines.

    Notation:
        C12 &= term_1 + term_2 \\
            &\quad + term_3 + term_4

    Examples:
        If the expression is shorter than lineWidth, this emits one line.
        If it is longer, terms are greedily wrapped at term boundaries.
    """
    if not expr:
        return f"{name} &= 0"

    terms = sortedTerms(expr)
    firstPrefix = f"{name} &= "
    nextPrefix = "&\\quad "

    lines = []
    current = firstPrefix

    for i, term in enumerate(terms):
        signed = latexTermSigned(
            term,
            i == 0,
        )

        if len(current + signed) > lineWidth and current not in (firstPrefix, nextPrefix):
            lines.append(current)
            current = nextPrefix + latexTermSigned(
                term,
                False,
            ).lstrip()
        else:
            current += signed

    lines.append(current)

    return " \\\\\n".join(lines)

def latexEquation(lhs: str, expr: Expr, lineWidth: int | None = None) -> str:
    """Render one numbered equation with optional split wrapping.

    Notation:
        \begin{equation}
        \begin{split}
        S_{\mathbb{CA}\rightarrow\mathbb{AV}} &= ...
        \end{split}
        \end{equation}
    """
    if lineWidth is None:
        body = f"{lhs} &= {latexExpr(expr)}"
    else:
        body = latexExprMultiline(
            lhs,
            expr,
            lineWidth,
        )

    return (
        "\\begin{equation}\n"
        "\\begin{split}\n"
        f"{body}\n"
        "\\end{split}\n"
        "\\end{equation}"
    )

def overlapLatexName(name: str) -> str:
    """Return printed overlap-metric label."""
    return overlapBlock(name).latexName

def residualLatexName(name: str) -> str:
    """Return printed zeroth-order residual label."""
    spec = EXCITATIONS[name]
    upper = latexIndexTuple(spec.creators)
    lower = latexIndexTuple(spec.annihilators)

    return rf"R^{{{upper}}}_{{{lower},(0)}}"
