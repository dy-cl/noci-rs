from __future__ import annotations

import argparse

from equations import outputExpr
from latex import latexEquation, overlapLatexName
from specs import availableBlocks

def availableBlockArgs() -> tuple[str, ...]:
    return availableBlocks() + ("all",)

def emitBlock(name: str, emit: str, lineWidth: int | None = None) -> str:
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
    """Emit one block or all overlap blocks."""
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
