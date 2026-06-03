from __future__ import annotations

import argparse

from core import (
    Idx,
    Product,
    Ref,
    Space,
    Wick,
    groupE2,
    tau1,
)
from latex import latexEquation
from canonical import canonicaliseForOutput

def A(name: str) -> Idx:
    return Idx(name, Space.ACTIVE)


def C(name: str) -> Idx:
    return Idx(name, Space.CORE)


def V(name: str) -> Idx:
    return Idx(name, Space.VIRTUAL)

def availableBlocks() -> tuple[str, ...]:
    return tuple(
        f"C{i}"
        for i in range(1, 20)
    )

def availableBlockArgs() -> tuple[str, ...]:
    return availableBlocks() + ("all",)

def transitionLabel(left: str, right: str) -> str:
    return f"\\mathbb{{{left}}}\\rightarrow\\mathbb{{{right}}}"


def blockLatexName(name: str) -> str:
    """Return printed overlap-metric label.

    Notation:
        C4 -> S_{\mathbb{CA}\rightarrow\mathbb{AV}}

    Mixed blocks use two transition labels:

        C13 -> S_{\mathbb{A}\rightarrow\mathbb{V},\,
                 \mathbb{AA}\rightarrow\mathbb{AV}}
    """
    labels = {
        "C1": ("C", "A"),
        "C2": ("A", "V"),
        "C3": ("A", "A"),
        "C4": ("CA", "AV"),
        "C5": ("CA", "VA"),
        "C6": ("CA", "VV"),
        "C7": ("CC", "AV"),
        "C8": ("CC", "AA"),
        "C9": ("CA", "AA"),
        "C10": ("AA", "AV"),
        "C11": ("AA", "VV"),
        "C12": ("AA", "AA"),
        "C17": ("C", "V"),
    }

    mixedLabels = {
        "C13": (("A", "V"), ("AA", "AV")),
        "C14": (("C", "A"), ("CA", "AA")),
        "C15": (("A", "A"), ("AA", "AA")),
        "C16": (("CA", "AV"), ("CA", "VA")),
        "C18": (("C", "V"), ("CA", "AV")),
        "C19": (("C", "V"), ("CA", "VA")),
    }

    if name in labels:
        left, right = labels[name]
        return f"S_{{{transitionLabel(left, right)}}}"

    if name in mixedLabels:
        first, second = mixedLabels[name]
        return (
            "S_{"
            + transitionLabel(*first)
            + ",\\,"
            + transitionLabel(*second)
            + "}"
        )

    raise ValueError(f"unknown overlap block {name}")

def blockProduct(name: str) -> Product:
    """Build the GNO operator product defining one overlap block.

    Notation:
        C_n = <Phi| tau_left^\dagger tau_right |Phi>

    Examples:
        C1 uses {E^i_u}{E^v_j}
        C3 uses {E^u_v}{E^x_w}
    """
    if name == "C1":
        i = C("i")
        j = C("j")
        u = A("u")
        v = A("v")

        return Product((
            tau1(i, u, 0),
            tau1(v, j, 1),
        ))

    if name == "C2":
        a = V("a")
        b = V("b")
        t = A("t")
        u = A("u")

        return Product((
            tau1(t, a, 0),
            tau1(b, u, 1),
        ))

    if name == "C3":
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            tau1(u, v, 0),
            tau1(x, w, 1),
        ))

    if name == "C4":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            groupE2(i, u, v, a, 0),
            groupE2(x, b, j, w, 1),
        ))

    if name == "C5":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            groupE2(i, u, a, v, 0),
            groupE2(b, x, j, w, 1),
        ))

    if name == "C6":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        c = V("c")
        d = V("d")
        u = A("u")
        v = A("v")

        return Product((
            groupE2(i, u, a, b, 0),
            groupE2(c, d, j, v, 1),
        ))

    if name == "C7":
        i = C("i")
        j = C("j")
        k = C("k")
        l = C("l")
        a = V("a")
        b = V("b")
        u = A("u")
        v = A("v")

        return Product((
            groupE2(i, j, u, a, 0),
            groupE2(v, b, k, l, 1),
        ))

    if name == "C8":
        i = C("i")
        j = C("j")
        k = C("k")
        l = C("l")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            groupE2(i, j, u, v, 0),
            groupE2(w, x, k, l, 1),
        ))

    if name == "C9":
        i = C("i")
        j = C("j")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")
        y = A("y")
        z = A("z")

        return Product((
            groupE2(i, u, v, w, 0),
            groupE2(y, z, j, x, 1),
        ))

    if name == "C10":
        a = V("a")
        b = V("b")
        t = A("t")
        u = A("u")
        v = A("v")
        x = A("x")
        y = A("y")
        z = A("z")

        return Product((
            groupE2(t, u, v, a, 0),
            groupE2(z, b, x, y, 1),
        ))

    if name == "C11":
        a = V("a")
        b = V("b")
        c = V("c")
        d = V("d")
        t = A("t")
        u = A("u")
        v = A("v")
        w = A("w")

        return Product((
            groupE2(t, u, a, b, 0),
            groupE2(c, d, v, w, 1),
        ))

    if name == "C12":
        p = A("p")
        q = A("q")
        r = A("r")
        s = A("s")
        t = A("t")
        u = A("u")
        v = A("v")
        w = A("w")

        return Product((
            groupE2(p, r, q, s, 0),
            groupE2(t, v, u, w, 1),
        ))

    if name == "C13":
        a = V("a")
        b = V("b")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            tau1(u, a, 0),
            groupE2(x, b, v, w, 1),
        ))

    if name == "C14":
        i = C("i")
        j = C("j")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            tau1(i, u, 0),
            groupE2(w, x, j, v, 1),
        ))

    if name == "C15":
        t = A("t")
        u = A("u")
        w = A("w")
        x = A("x")
        y = A("y")
        z = A("z")

        return Product((
            tau1(t, u, 0),
            groupE2(y, z, w, x, 1),
        ))

    if name == "C16":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        u = A("u")
        w = A("w")
        x = A("x")
        y = A("y")

        return Product((
            groupE2(i, u, w, a, 0),
            groupE2(b, y, j, x, 1),
        ))

    if name == "C17":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")

        return Product((
            tau1(i, a, 0),
            tau1(b, j, 1),
        ))

    if name == "C18":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        w = A("w")
        x = A("x")

        return Product((
            tau1(i, a, 0),
            groupE2(x, b, j, w, 1),
        ))

    if name == "C19":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        w = A("w")
        x = A("x")

        return Product((
            tau1(i, a, 0),
            groupE2(b, x, j, w, 1),
        ))

    raise ValueError(f"unknown overlap block {name}")

def overlapExpr(name: str):
    """Evaluate one overlap block by Wick algebra."""
    return Wick(Ref()).eval(
        blockProduct(name)
    )

def outputExpr(name: str):
    """Evaluate and canonicalise one overlap block for emission."""
    return canonicaliseForOutput(
        overlapExpr(name)
    )

def emitBlock(name: str, emit: str, lineWidth: int | None = None) -> str:
    expr = outputExpr(name)

    if emit == "latex":
        return latexEquation(
            blockLatexName(name),
            expr,
            lineWidth = lineWidth,
        )

    if emit == "expr":
        return repr(expr)

    raise ValueError(f"unknown emit mode {emit}")

def emitBlocks(name: str, emit: str, lineWidth: int | None = None) -> str:
    """Emit one block or all overlap blocks.

    Notation:
        --block C1   emits C1
        --block all  emits C1 ... C16
    """
    if name == "all":
        return "\n\n".join(
            emitBlock(
                block,
                emit,
                lineWidth = lineWidth,
            )
            for block in availableBlocks()
        )

    return emitBlock(
        name,
        emit,
        lineWidth = lineWidth,
    )

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--block",
        choices = availableBlockArgs(),
        required = True,
    )

    parser.add_argument(
        "--emit",
        choices = ("latex", "expr"),
        default = "latex",
    )

    parser.add_argument(
        "--line-width",
        type = int,
        default = 120,
        help = "maximum LaTeX line width before wrapping; use 0 to disable wrapping",
    )

    args = parser.parse_args()

    lineWidth = None if args.line_width <= 0 else args.line_width

    print(
        emitBlocks(
            args.block,
            args.emit,
            lineWidth = lineWidth,
        )
    )

if __name__ == "__main__":
    main()
