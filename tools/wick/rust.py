# tools/wick/rust.py

from __future__ import annotations

from fractions import Fraction

from symbols import (
    Expr,
    Idx,
    Delta,
    Tensor,
    Term,
)


def rustIdx(i: Idx) -> str:
    """Return the Rust variable name for an index."""

    return i.name


def rustDelta(d: Delta) -> str:
    """Emit a Kronecker delta factor."""

    return f"delta({rustIdx(d.left)}, {rustIdx(d.right)})"


def rustActiveIndex(i: Idx) -> str:
    """Emit active-space index lookup for cumulant tensors."""

    return f"spaces.active_map[{rustIdx(i)}].unwrap()"


def rustGamma1(t: Tensor) -> str:
    """Emit a Gamma1 access.

    Gamma^p_q -> gamma1.data[p * gamma1.n + q]
    """

    p = rustIdx(t.upper[0])
    q = rustIdx(t.lower[0])

    return f"gamma1.data[{p} * gamma1.n + {q}]"


def rustTheta(t: Tensor) -> str:
    """Emit a Theta access.

    Theta^p_q -> theta(gamma1, p, q)
    """

    p = rustIdx(t.upper[0])
    q = rustIdx(t.lower[0])

    return f"theta(gamma1, {p}, {q})"


def rustLambda(t: Tensor) -> str:
    """Emit a Lambda2/Lambda3/Lambda4 access."""

    if t.name == "Lambda2":
        field = "lambda2"
    elif t.name == "Lambda3":
        field = "lambda3"
    elif t.name == "Lambda4":
        field = "lambda4"
    else:
        raise ValueError(f"Cannot emit {t.name} as a Lambda tensor.")

    upper = ", ".join(rustActiveIndex(i) for i in t.upper)
    lower = ", ".join(rustActiveIndex(i) for i in t.lower)

    return (
        f"lambdas.{field}.get(\n"
        f"    &[{upper}],\n"
        f"    &[{lower}],\n"
        f")"
    )


def rustTensor(t: Tensor) -> str:
    """Emit a scalar tensor factor."""

    if t.name == "Gamma1":
        return rustGamma1(t)

    if t.name == "Theta":
        return rustTheta(t)

    if t.name in {"Lambda2", "Lambda3", "Lambda4"}:
        return rustLambda(t)

    raise ValueError(f"Cannot emit tensor {t.name} to final Rust.")


def rustCoeff(c: Fraction) -> str | None:
    """Emit a positive coefficient.

    Returns None for coefficient 1.
    """

    if c < 0:
        raise ValueError("rustCoeff expects a non-negative coefficient.")

    if c == 1:
        return None

    if c.denominator == 1:
        return f"{c.numerator}.0"

    if c.denominator in {2, 4, 8, 16}:
        return str(float(c))

    return f"({c.numerator}.0 / {c.denominator}.0)"


def rustTermFactors(t: Term) -> list[str]:
    """Emit all multiplicative factors in a term."""

    factors = []

    coeff = rustCoeff(abs(t.coeff))

    if coeff is not None:
        factors.append(coeff)

    for d in t.deltas:
        factors.append(rustDelta(d))

    for tensor in t.tensors:
        factors.append(rustTensor(tensor))

    if t.generators:
        raise ValueError("Cannot emit unresolved E1 generators to Rust.")

    if not factors:
        factors.append("1.0")

    return factors


def indentMultilineFactor(factor: str, indent: str) -> str:
    """Indent continuation lines in a multiline Rust factor."""

    lines = factor.splitlines()

    if len(lines) == 1:
        return factor

    return lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])


def rustTerm(t: Term) -> str:
    """Emit one Rust accumulation statement."""

    factors = rustTermFactors(t)

    if len(factors) == 1:
        expr = factors[0]
    else:
        first = indentMultilineFactor(factors[0], "    ")
        rest = [
            indentMultilineFactor(f, "    ")
            for f in factors[1:]
        ]
        expr = first + "\n    * " + "\n    * ".join(rest)

    if t.coeff < 0:
        return f"out -= {expr};"

    return f"out += {expr};"


def rustExpr(expr: Expr) -> str:
    """Emit straight-line Rust accumulation statements."""

    if not expr:
        return "let out = 0.0;\n\nout"

    lines = ["let mut out = 0.0;"]

    for t in expr:
        lines.append("")
        lines.append(rustTerm(t))

    lines.append("")
    lines.append("out")

    return "\n".join(lines)


def specVariables(spec) -> str:
    """Return Rust destructuring code for the left excitation spec."""

    from classes import SingleSpec, DoubleSpec

    if isinstance(spec, SingleSpec):
        p = rustIdx(spec.create)
        q = rustIdx(spec.annihilate)

        return f"let ({p}, {q}) = single(left);"

    if isinstance(spec, DoubleSpec):
        p = rustIdx(spec.create1)
        q = rustIdx(spec.create2)
        r = rustIdx(spec.annihilate1)
        s = rustIdx(spec.annihilate2)

        return f"let ({p}, {q}, {r}, {s}) = double(left);"

    raise TypeError(f"Unsupported excitation spec {type(spec)}")


def rightSpecVariables(spec) -> str:
    """Return Rust destructuring code for the right excitation spec."""

    from classes import SingleSpec, DoubleSpec

    if isinstance(spec, SingleSpec):
        p = rustIdx(spec.create)
        q = rustIdx(spec.annihilate)

        return f"let ({p}, {q}) = single(right);"

    if isinstance(spec, DoubleSpec):
        p = rustIdx(spec.create1)
        q = rustIdx(spec.create2)
        r = rustIdx(spec.annihilate1)
        s = rustIdx(spec.annihilate2)

        return f"let ({p}, {q}, {r}, {s}) = double(right);"

    raise TypeError(f"Unsupported excitation spec {type(spec)}")

def rustFunction(block, expr: Expr) -> str:
    """Emit a generated Rust overlap function."""

    body = rustExpr(expr)

    indentedBody = "\n".join(
        "    " + line if line else ""
        for line in body.splitlines()
    )

    leftLine = specVariables(block.left)
    rightLine = rightSpecVariables(block.right)

    return (
        f"/// Generated overlap block {block.name}: {block.label}.\n"
        "/// WARNING: This function is generated by `tools/wick` and should not be edited by hand.\n"
        "/// Evaluates `<Phi | tau_left^dagger tau_right | Phi>` for the given spin-free excitations.\n"
        "/// # Arguments:\n"
        "/// - `left`: Raw left spin-free excitation.\n"
        "/// - `right`: Raw right spin-free excitation.\n"
        "/// - `spaces`: Core, active and virtual orbital-space metadata.\n"
        "/// - `gamma1`: Full-space spin-free one-body RDM.\n"
        "/// - `lambdas`: Active-space spin-free cumulants.\n"
        "/// # Returns:\n"
        "/// - `f64`: Spin-free overlap metric element.\n"
        f"fn {block.functionName}(\n"
        "    left: Excitation,\n"
        "    right: Excitation,\n"
        "    spaces: &Spaces,\n"
        "    gamma1: &RDM1<f64>,\n"
        "    lambdas: &Cumulants<f64>,\n"
        ") -> f64 {\n"
        f"    {leftLine}\n"
        f"    {rightLine}\n\n"
        f"{indentedBody}\n"
        "}\n"
    )
