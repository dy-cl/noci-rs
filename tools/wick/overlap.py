from __future__ import annotations

import argparse

from equations import outputExpr
from latex import latexEquation, overlapLatexName
from specs import availableBlocks

def availableBlockArgs() -> tuple[str, ...]:
    """
    Return valid command-line overlap block arguments.

    This includes every overlap block defined in specs.py, plus the special
    value "all".

    Notation:
        C_1, C_2, ..., C_k, all

    Examples:
        availableBlockArgs() returns  ("C1", "C2", ..., "C19", "all").
    """
    return availableBlocks() + ("all",)

def emitBlock(name: str, emit: str, lineWidth: int | None = None) -> str:
    """
    Emit one overlap block in the requested format.

    Notation:
        S_{\mu\nu} = \langle \Phi | \tau_\mu^\dagger \tau_\nu | \Phi \rangle

    Examples:
        emitBlock("C4", "latex") emits the LaTeX equation for C4.
        emitBlock("C4", "expr") emits the Python repr of the symbolic Expr.
        emitBlock("C4", "rust") emits the Rust function for C4.
    """
    if emit == "rust":
        from rust import rustOverlapFunction

        return rustOverlapFunction(name)

    expr = outputExpr(name)

    if emit == "latex":
        return latexEquation(
            overlapLatexName(name),
            expr,
            lineWidth = lineWidth,
        )

    if emit == "expr":
        return repr(expr)

    raise ValueError(f"unknown emit mode {emit}")

def emitBlocks(name: str, emit: str, lineWidth: int | None = None) -> str:
    """
    Emit one overlap block or all overlap blocks.

    Notation:
        S_{\mu\nu} = \langle \Phi | \tau_\mu^\dagger \tau_\nu | \Phi \rangle

    Examples:
        emitBlocks("C1", "latex") emits only C1.
        emitBlocks("all", "latex") emits all overlap block equations.
        emitBlocks("all", "rust") emits the complete generated Rust module.
    """
    if name == "all" and emit == "rust":
        from rust import rustModule

        return rustModule()

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
    """
    Run the overlap block generator CLI.

    Notation:

    Examples:
        python tools/wick/overlap.py --block C4 --emit latex
        python tools/wick/overlap.py --block C4 --emit expr
        python tools/wick/overlap.py --block all --emit rust
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--block",
        choices = availableBlockArgs(),
        required = True,
    )

    parser.add_argument(
        "--emit",
        choices = ("latex", "expr", "rust"),
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
        ),
        end = "" if args.emit == "rust" and args.block == "all" else "\n",
    )

if __name__ == "__main__":
    main()
