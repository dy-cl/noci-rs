# tools/wick/overlap.py

from __future__ import annotations

import argparse

from symbols import Expr, mul
from generators import tau1, tau2
from wick import expectation
from cumulants import rewriteGammaToLambda
from latex import latexEquation
from rust import rustFunction
from classes import (
    SingleSpec,
    DoubleSpec,
    OverlapBlock,
    blockByName,
    availableBlocks,
)


def excitationExpr(spec: SingleSpec | DoubleSpec) -> Expr:
    """Build the raw GNO excitation expression for a spec."""

    if isinstance(spec, SingleSpec):
        return tau1(
            spec.create,
            spec.annihilate,
        )

    if isinstance(spec, DoubleSpec):
        return tau2(
            spec.create1,
            spec.create2,
            spec.annihilate1,
            spec.annihilate2,
        )

    raise TypeError(f"Unsupported excitation spec {type(spec)}")

def gammaSummary(expr: Expr) -> str:
    """Return a summary of unresolved Gamma tensors."""

    counts = {}

    for termIn in expr:
        for tensorIn in termIn.tensors:
            if not tensorIn.name.startswith("Gamma"):
                continue

            spaces = (
                tuple(i.space.value for i in tensorIn.upper),
                tuple(i.space.value for i in tensorIn.lower),
            )

            key = (tensorIn.name, spaces)
            counts[key] = counts.get(key, 0) + 1

    lines = []

    for (name, spaces), count in sorted(counts.items()):
        upperSpaces, lowerSpaces = spaces
        lines.append(
            f"{count:4d} {name} upper={''.join(upperSpaces)} lower={''.join(lowerSpaces)}"
        )

    return "\n".join(lines)

def adjointExcitationExpr(spec: SingleSpec | DoubleSpec) -> Expr:
    """Build the daggered GNO excitation expression for a raw spec.

    For singles:

        (tau^p_q)^dagger = tau^q_p

    For doubles:

        (tau^{pq}_{rs})^dagger = tau^{rs}_{pq}
    """

    if isinstance(spec, SingleSpec):
        return tau1(
            spec.annihilate,
            spec.create,
        )

    if isinstance(spec, DoubleSpec):
        return tau2(
            spec.annihilate1,
            spec.annihilate2,
            spec.create1,
            spec.create2,
        )

    raise TypeError(f"Unsupported excitation spec {type(spec)}")

def rawOverlapExpr(block: OverlapBlock) -> Expr:
    """Generate the unreduced overlap expression for a block.

    Computes:

        <Phi | tau_left^dagger tau_right | Phi>
    """

    raw = mul(
        adjointExcitationExpr(block.left),
        excitationExpr(block.right),
    )

    return expectation(raw)


def overlapExpr(block: OverlapBlock) -> Expr:
    """Generate the reduced overlap expression for a block."""

    return rewriteGammaToLambda(
        rawOverlapExpr(block)
    )

def blockLhs(block: OverlapBlock) -> str:
    """Return a simple LaTeX left-hand side for a block."""

    return block.name

def emitBlock(block: OverlapBlock, mode: str) -> str:
    """Emit a block in the requested format."""

    if mode == "raw-latex":
        return latexEquation(
            blockLhs(block),
            rawOverlapExpr(block),
        )

    if mode == "raw-expr":
        return str(rawOverlapExpr(block))

    if mode == "raw-gammas":
        return gammaSummary(rawOverlapExpr(block))

    expr = overlapExpr(block)

    if mode == "latex":
        return latexEquation(
            blockLhs(block),
            expr,
        )

    if mode == "rust":
        return rustFunction(
            block,
            expr,
        )

    if mode == "expr":
        return str(expr)

def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Generate spin-free overlap expressions."
    )

    parser.add_argument(
        "--block",
        choices = availableBlocks(),
        required = True,
        help = "Overlap block to generate.",
    )

    parser.add_argument(
        "--emit",
        choices = ["latex", "rust", "expr", "raw-latex", "raw-expr", "raw-gammas"],
        default = "latex",
        help = "Output format.",
    )

    args = parser.parse_args()

    block = blockByName(args.block)
    print(emitBlock(block, args.emit))


if __name__ == "__main__":
    main()
