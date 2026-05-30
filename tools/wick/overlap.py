# tools/wick/overlap.py

from __future__ import annotations

import argparse

from symbols import Expr, Space, mul
from generators import tau1, tau2
from wick import expectation
from cumulants import rewriteGammaToLambda
from gno import gnoOverlapBlock
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
    """Build the raw expanded GNO excitation expression for a spec."""
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
    """Build the daggered expanded GNO excitation expression for a raw spec."""
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
    """Generate the unreduced expanded Wick-route overlap expression."""
    raw = mul(
        adjointExcitationExpr(block.left),
        excitationExpr(block.right),
    )

    return expectation(raw)


def specIndices(spec: SingleSpec | DoubleSpec) -> tuple:
    """Return all indices appearing in an excitation spec."""
    if isinstance(spec, SingleSpec):
        return (
            spec.create,
            spec.annihilate,
        )

    return (
        spec.create1,
        spec.create2,
        spec.annihilate1,
        spec.annihilate2,
    )


def blockIndices(block: OverlapBlock) -> tuple:
    """Return all orbital indices appearing in a block."""
    return specIndices(block.left) + specIndices(block.right)


def allBlockIndicesActive(block: OverlapBlock) -> bool:
    """Return True if every index in a block is active."""
    return all(
        index.space == Space.ACTIVE
        for index in blockIndices(block)
    )


def overlapExpr(block: OverlapBlock) -> Expr:
    """Generate the reduced overlap expression for a block.

    All-active blocks still use the older Gamma-to-Lambda route because that
    route contains the validated Lambda3 spin-free reconstruction. Mixed C/A/V
    blocks use the grouped GNO evaluator, which keeps normal-order group
    boundaries and applies generalized Wick algebra directly.
    """
    if allBlockIndicesActive(block):
        return rewriteGammaToLambda(
            rawOverlapExpr(block)
        )

    return gnoOverlapBlock(block)


def blockLhs(block: OverlapBlock) -> str:
    """Return a simple LaTeX left-hand side for a block."""
    return block.name


def termOriginLabel(termIn) -> str:
    """Return a compact label for a raw overlap term."""
    gens = " ".join(
        f"E({g.upper.name},{g.lower.name})"
        for g in termIn.generators
    )
    tensors = " ".join(
        f"{t.name}({''.join(i.name for i in t.upper)};{''.join(i.name for i in t.lower)})"
        for t in termIn.tensors
    )
    deltas = " ".join(
        f"d({d.left.name},{d.right.name})"
        for d in termIn.deltas
    )

    return f"coeff={termIn.coeff} deltas=[{deltas}] tensors=[{tensors}] gens=[{gens}]"


def printRawOverlapTerms(block: OverlapBlock) -> str:
    """Return raw Wick-route terms for debugging algebra rewrites."""
    lines = []

    for termIn in rawOverlapExpr(block):
        lines.append(termOriginLabel(termIn))

    return "\n".join(lines)


def emitBlock(block: OverlapBlock, mode: str) -> str:
    """Emit a block in the requested format."""
    if mode == "raw-terms":
        return printRawOverlapTerms(block)

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

    raise ValueError(f"Unsupported emit mode {mode}")


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
        choices = ["latex", "rust", "expr", "raw-latex", "raw-expr", "raw-gammas", "raw-terms"],
        default = "latex",
        help = "Output format.",
    )

    args = parser.parse_args()
    block = blockByName(args.block)

    print(emitBlock(block, args.emit))


if __name__ == "__main__":
    main()
