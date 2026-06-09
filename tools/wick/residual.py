from __future__ import annotations

import argparse

from equations import residualExpr
from latex import latexEquation, residualLatexName
from specs import availableExcitations

def emitResidual(name: str, emit: str, order: int = 0, lineWidth: int | None = None) -> str:
    """
    Emit one residual contribution.

    Notation:
        R_mu^{(n)}

    Examples:
        emitResidual("CToA", "expr", order = 0) emits the direct residual.
        emitResidual("CToA", "expr", order = 1) emits the first-order residual.
    """
    if emit == "rust":
        if order != 0:
            raise ValueError("Rust emission is currently only implemented for zeroth-order residuals")

        from rust import rustResidualFunction

        return rustResidualFunction(name)

    expr = residualExpr(
        name,
        order = order,
    )

    if emit == "latex":
        return latexEquation(
            residualLatexName(name, order = order),
            expr,
            lineWidth = lineWidth,
        )

    if emit == "expr":
        return repr(expr)

    raise ValueError(f"unknown emit mode {emit}")

def emitResiduals(name: str, emit: str, order: int = 0, lineWidth: int | None = None) -> str:
    """
    Emit one residual expression or all residual expressions.

    Notation:

    Examples:
        emitResiduals("CtoA", "latex") emits only the C -> A residual.
        emitResiduals("all", "latex") emits all residual equations.
        emitResiduals("all", "rust") emits the complete generated Rust module.
    """
    if name == "all" and emit == "rust":
        from rust import rustResidualModule

        if order != 0:
            raise ValueError("Rust emission is currently only implemented for zeroth-order residuals")

        return rustResidualModule()

    if name == "all":
        return "\n\n".join(
            emitResidual(
                excitation,
                emit,
                order = order,
                lineWidth = lineWidth,
            )
            for excitation in availableExcitations()
        )

    return emitResidual(
        name,
        emit,
        order = order,
        lineWidth = lineWidth,
    )

def main() -> None:
    """
    Run the zeroth-order residual generator CLI.

    Notation:

    Examples:
        python tools/wick/residual.py --class CtoA --emit latex
        python tools/wick/residual.py --class CtoA --emit expr
        python tools/wick/residual.py --class all --emit rust
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--class",
        dest = "excitation",
        choices = availableExcitations() + ("all",),
        required = True,
    )

    parser.add_argument(
        "--emit",
        choices = ("latex", "expr", "rust"),
        default = "latex",
    )

    parser.add_argument(
        "--order",
        type = int,
        choices = (0, 1),
        default = 0,
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
        emitResiduals(
            args.excitation,
            args.emit,
            order = args.order,
            lineWidth = lineWidth,
        ),
        end = "" if args.emit == "rust" and args.excitation == "all" else "\n",
    )

if __name__ == "__main__":
    main()
