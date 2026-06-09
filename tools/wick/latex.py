from __future__ import annotations

from fractions import Fraction

from core import Delta, Expr, Idx, Tensor, Term
from specs import EXCITATIONS, overlapBlock

def latexIndex(idx: Idx) -> str:
    """
    Render one orbital index.

    Notation:
        i, j, k, l \in C
        u, v, w, x \in A
        a, b, c, d \in V

    Examples:
        latexIndex(i) gives "i".

        latexIndex(u) gives "u".
    """
    return idx.name

def latexIndexTuple(indices: tuple[Idx, ...]) -> str:
    """
    Render an upper or lower tensor index tuple.

    Notation:
        (u, x, v) -> uxv

    Examples:
        latexIndexTuple((u, x)) gives "ux".
        latexIndexTuple((p, q, r, s)) gives "pqrs".
    """
    return "".join(
        latexIndex(idx)
        for idx in indices
    )

def latexDelta(delta: Delta) -> str:
    """
    Render a Kronecker delta.

    Notation:
        \delta^p_q

    Examples:
        Delta(i, j) becomes \delta^{i}_{j}.
        Delta(a, b) becomes \delta^{a}_{b}.
    """
    return (
        "\\delta"
        + f"^{{{latexIndex(delta.left)}}}"
        + f"_{{{latexIndex(delta.right)}}}"
    )

def latexTensor(tensor: Tensor) -> str:
    """
    Render a spin-free tensor.

    Notation:
        Gamma1  -> \Gamma
        Theta   -> \Theta
        Lambda2 -> \Lambda
        Lambda3 -> \Lambda
        Lambda4 -> \Lambda

    Examples:
        Tensor("Gamma1", (u,), (v,)) becomes \Gamma^{u}_{v}.

        Tensor("Theta", (x,), (w,)) becomes \Theta^{x}_{w}.

        Tensor("Lambda2", (u, x), (v, w)) becomes \Lambda^{ux}_{vw}.
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
    """
    Render an absolute coefficient. The sign is handled outside 
    this function, so this only renders |coeff|.

    Notation:
        1 ---> omitted if there are tensor/delta factors
        1/2 ---> \frac{1}{2}
        3 ---> 3

    Examples:
        latexCoeff(Fraction(1), True) gives "".
        latexCoeff(Fraction(1, 2), True) gives \frac{1}{2}.
        latexCoeff(Fraction(3), False) gives "3".
    """
    c = abs(coeff)

    if c == 1 and hasFactors:
        return ""

    if c.denominator == 1:
        return str(c.numerator)

    return f"\\frac{{{c.numerator}}}{{{c.denominator}}}"

def latexTermBody(term: Term) -> str:
    """
    Render one term without its leading sign.

    Notation:
        c \delta \Gamma \Lambda

    Examples:
        Term(
            coeff = 1/2,
            deltas = (\delta^i_j,),
            tensors = (\Gamma^u_w, \Theta^x_v),
        )

        becomes \frac{1}{2} \delta^{i}_{j} \Gamma^{u}_{w} \Theta^{x}_{v}.
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
    """
    Render one signed term. This adds the leading sign appropriate for a term in a sum.

    Notation:

    Examples:
        First positive term: \Gamma^{u}_{v}
        Later positive term: + \Lambda^{ux}_{vw}
        Negative term: - \frac{1}{2} \Gamma^{u}_{v}\Gamma^{x}_{w}
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
    """
    Return terms in stable display order.
    This makes repeated equation generation produce stable text output.

    Notation:
        Sort by deltas, tensors, then coefficient.

    Examples:
        Terms with the same deltas and tensor factors appear together in a
        deterministic order.

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
    """
    Render a complete expression on one line.

    Notation:

    Examples:
        Two terms become \Gamma^{u}_{v} + \Lambda^{ux}_{vw}.
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
    """
    Render an expression over multiple split lines.

    Notation:
        C_{12} &= term_1 + term_2 \\
               &\quad + term_3 + term_4

    Examples:
        If the expression is shorter than lineWidth, this emits one line.
        If it is longer, terms are wrapped at term boundaries.
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
    """
    Render one numbered LaTeX equation with optional split wrapping.

    Notation:
        \begin{equation}
        \begin{split}
        S_{\mathbb{CA}\rightarrow\mathbb{AV}} &= ...
        \end{split}
        \end{equation}

    Examples:
        latexEquation("C_1", expr) emits one equation.
        latexEquation("C_{12}", expr, lineWidth = 100) emits a wrapped
        split equation if needed.
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
    """
    Return the printed overlap-metric label.

    Notation:
        C_k
        S_{\mu\nu}

    Examples:
        overlapLatexName("C4") returns the configured LaTeX label for the C4 overlap block.
    """
    return overlapBlock(name).latexName

def residualLatexName(name: str) -> str:
    """
    Return the printed zeroth-order residual label.

    Notation:
        R^{p_1 \cdots p_k}_{q_1 \cdots q_k,(0)}

    Examples:
        For an excitation with creators (u,) and annihilators (i,), this
        returns R^{u}_{i,(0)}.

        For an excitation with creators (u, a) and annihilators (i, v), this
        returns R^{ua}_{iv,(0)}.
    """
    spec = EXCITATIONS[name]
    upper = latexIndexTuple(spec.creators)
    lower = latexIndexTuple(spec.annihilators)

    return rf"R^{{{upper}}}_{{{lower},(0)}}"
