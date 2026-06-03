from __future__ import annotations

from dataclasses import dataclass
import argparse

from core import Delta, Expr, Tensor, Term
from overlap import overlapExpr, outputExpr


@dataclass(frozen = True)
class RustBlock:
    block: str
    function: str
    left_class: str
    right_class: str
    left_unpack: tuple[str, ...]
    right_unpack: tuple[str, ...]


BLOCKS = (
    RustBlock(
        "C1",
        "overlap_c_to_a_c_to_a",
        "CToA",
        "CToA",
        ("u", "i"),
        ("v", "j"),
    ),
    RustBlock(
        "C2",
        "overlap_a_to_v_a_to_v",
        "AToV",
        "AToV",
        ("a", "t"),
        ("b", "u"),
    ),
    RustBlock(
        "C3",
        "overlap_a_to_a_a_to_a",
        "AToA",
        "AToA",
        ("v", "u"),
        ("x", "w"),
    ),
    RustBlock(
        "C4",
        "overlap_ca_to_av_ca_to_av",
        "CAToAV",
        "CAToAV",
        ("v", "a", "i", "u"),
        ("x", "b", "j", "w"),
    ),
    RustBlock(
        "C5",
        "overlap_ca_to_va_ca_to_va",
        "CAToVA",
        "CAToVA",
        ("a", "v", "i", "u"),
        ("b", "x", "j", "w"),
    ),
    RustBlock(
        "C6",
        "overlap_ca_to_vv_ca_to_vv",
        "CAToVV",
        "CAToVV",
        ("a", "b", "i", "u"),
        ("c", "d", "j", "v"),
    ),
    RustBlock(
        "C7",
        "overlap_cc_to_av_cc_to_av",
        "CCToAV",
        "CCToAV",
        ("u", "a", "i", "j"),
        ("v", "b", "k", "l"),
    ),
    RustBlock(
        "C8",
        "overlap_cc_to_aa_cc_to_aa",
        "CCToAA",
        "CCToAA",
        ("u", "v", "i", "j"),
        ("w", "x", "k", "l"),
    ),
    RustBlock(
        "C9",
        "overlap_ca_to_aa_ca_to_aa",
        "CAToAA",
        "CAToAA",
        ("v", "w", "i", "u"),
        ("y", "z", "j", "x"),
    ),
    RustBlock(
        "C10",
        "overlap_aa_to_av_aa_to_av",
        "AAToAV",
        "AAToAV",
        ("v", "a", "t", "u"),
        ("z", "b", "x", "y"),
    ),
    RustBlock(
        "C11",
        "overlap_aa_to_vv_aa_to_vv",
        "AAToVV",
        "AAToVV",
        ("a", "b", "t", "u"),
        ("c", "d", "v", "w"),
    ),
    RustBlock(
        "C12",
        "overlap_aa_to_aa_aa_to_aa",
        "AAToAA",
        "AAToAA",
        ("q", "s", "p", "r"),
        ("t", "v", "u", "w"),
    ),
    RustBlock(
        "C13",
        "overlap_a_to_v_aa_to_av",
        "AToV",
        "AAToAV",
        ("a", "u"),
        ("x", "b", "v", "w"),
    ),
    RustBlock(
        "C14",
        "overlap_c_to_a_ca_to_aa",
        "CToA",
        "CAToAA",
        ("u", "i"),
        ("w", "x", "j", "v"),
    ),
    RustBlock(
        "C15",
        "overlap_a_to_a_aa_to_aa",
        "AToA",
        "AAToAA",
        ("u", "t"),
        ("y", "z", "w", "x"),
    ),
    RustBlock(
        "C16",
        "overlap_ca_to_av_ca_to_va",
        "CAToAV",
        "CAToVA",
        ("w", "a", "i", "u"),
        ("b", "y", "j", "x"),
    ),
    RustBlock(
        "C17",
        "overlap_c_to_v_c_to_v",
        "CToV",
        "CToV",
        ("a", "i"),
        ("b", "j"),
    ),
    RustBlock(
        "C18",
        "overlap_c_to_v_ca_to_av",
        "CToV",
        "CAToAV",
        ("a", "i"),
        ("x", "b", "j", "w"),
    ),
    RustBlock(
        "C19",
        "overlap_c_to_v_ca_to_va",
        "CToV",
        "CAToVA",
        ("a", "i"),
        ("b", "x", "j", "w"),
    ),
)

def rustFunctionDoc(block: RustBlock) -> list[str]:
    return [
        f"/// Evaluate generated Appendix-C overlap block `{block.block}`.",
        "/// # Arguments:",
        "/// - `left`: Left excitation operator, daggered in the metric element.",
        "/// - `right`: Right excitation operator.",
        "/// - `spaces`: Core, active, and virtual orbital-space maps.",
        "/// - `gamma1`: Spin-free one-particle RDM.",
        "/// - `lambdas`: Spin-free active-space cumulants.",
        "/// # Returns:",
        "/// - `f64`: Raw FOIS overlap metric element.",
    ]

def blockByName(name: str) -> RustBlock:
    for block in BLOCKS:
        if block.block == name:
            return block

    raise ValueError(f"unknown block {name}")

def rustCoeff(coeff) -> str:
    if coeff.denominator == 1:
        return f"{coeff.numerator}.0"

    return f"({coeff.numerator}.0 / {coeff.denominator}.0)"

def rustDelta(delta: Delta) -> str:
    return f"delta({delta.left.name}, {delta.right.name})"

def rustTensor(tensor: Tensor) -> str:
    args = ", ".join(idx.name for idx in tensor.upper + tensor.lower)

    if tensor.name == "Gamma1":
        p = tensor.upper[0].name
        q = tensor.lower[0].name
        return f"gamma1.data[{p} * gamma1.n + {q}]"

    if tensor.name == "Theta":
        p = tensor.upper[0].name
        q = tensor.lower[0].name
        return f"theta(gamma1, {p}, {q})"

    if tensor.name == "Lambda2":
        upper = ", ".join(f"active(spaces, {idx.name})" for idx in tensor.upper)
        lower = ", ".join(f"active(spaces, {idx.name})" for idx in tensor.lower)
        return f"lambdas.lambda2.get(&[{upper}], &[{lower}])"

    if tensor.name == "Lambda3":
        upper = ", ".join(f"active(spaces, {idx.name})" for idx in tensor.upper)
        lower = ", ".join(f"active(spaces, {idx.name})" for idx in tensor.lower)
        return f"lambdas.lambda3.get(&[{upper}], &[{lower}])"

    if tensor.name == "Lambda4":
        upper = ", ".join(f"active(spaces, {idx.name})" for idx in tensor.upper)
        lower = ", ".join(f"active(spaces, {idx.name})" for idx in tensor.lower)
        return f"lambdas.lambda4.get(&[{upper}], &[{lower}])"

    raise ValueError(f"unknown tensor {tensor.name}")

def rustTerm(term: Term) -> str:
    factors = []

    if term.coeff != 1:
        factors.append(rustCoeff(term.coeff))

    factors.extend(
        rustDelta(delta)
        for delta in term.deltas
    )

    factors.extend(
        rustTensor(tensor)
        for tensor in term.tensors
    )

    if not factors:
        return "1.0"

    return " * ".join(factors)

def rustUnpack(names: tuple[str, ...], side: str) -> str:
    lhs = ", ".join(names)

    if len(names) == 2:
        return f"    let ({lhs}) = single({side});"

    if len(names) == 4:
        return f"    let ({lhs}) = double({side});"

    raise ValueError(f"unsupported unpack size {len(names)}")

def rustFunction(block: RustBlock, debug: bool = False) -> str:
    expr = outputExpr(block.block)

    lines = rustFunctionDoc(block) + [
        "#[allow(unused_variables)]",
        f"pub(crate) fn {block.function}(",
        "    left: Excitation,",
        "    right: Excitation,",
        "    spaces: &Spaces,",
        "    gamma1: &RDM1<f64>,",
        "    lambdas: &Cumulants<f64>,",
        ") -> f64 {",
        rustUnpack(block.left_unpack, "left"),
        rustUnpack(block.right_unpack, "right"),
    ]

    if not debug:
        lines.append("    let mut out = 0.0;")

        for term in expr:
            lines.append(f"    out += {rustTerm(term)};")

        lines.extend([
            "    out",
            "}",
        ])

        return "\n".join(lines)

    for i, term in enumerate(expr):
        lines.append(f"    let term_{i} = {rustTerm(term)};")

    if expr:
        lines.append(
            "    let out = "
            + " + ".join(f"term_{i}" for i in range(len(expr)))
            + ";"
        )
    else:
        lines.append("    let out = 0.0;")

    lines.extend([
        "    if left == right {",
        f'        eprintln!("DEBUG {block.block} {{:?}}", left);',
    ])

    for i in range(len(expr)):
        lines.append(f'        eprintln!("  term_{i}: {{:.16e}}", term_{i});')

    lines.extend([
        '        eprintln!("  total: {:.16e}", out);',
        "    }",
        "",
        "    out",
        "}",
    ])

    return "\n".join(lines)

def rustFunctionDebugTerms(block: RustBlock) -> str:
    expr = overlapExpr(block.block)

    lines = rustFunctionDoc(block) + [
        "#[allow(unused_variables)]",
        f"pub(crate) fn {block.function}(",
        "    left: Excitation,",
        "    right: Excitation,",
        "    spaces: &Spaces,",
        "    gamma1: &RDM1<f64>,",
        "    lambdas: &Cumulants<f64>,",
        ") -> f64 {",
        rustUnpack(block.left_unpack, "left"),
        rustUnpack(block.right_unpack, "right"),
    ]

    for i, term in enumerate(expr):
        lines.append(f"    let term_{i} = {rustTerm(term)};")

    terms = " + ".join(f"term_{i}" for i in range(len(expr))) or "0.0"
    lines.append(f"    let out = {terms};")

    lines.extend([
        f'    if left == right && matches!(excitation_class(spaces, left), ExcitationClass::{block.left_class}) {{',
        f'        eprintln!("DEBUG {block.block} {{:?}}", left);',
    ])

    for i in range(len(expr)):
        lines.append(f'        eprintln!("  term_{i}: {{:.16e}}", term_{i});')

    lines.extend([
        '        eprintln!("  total: {:.16e}", out);',
        "    }",
        "",
        "    out",
        "}",
    ])

    return "\n".join(lines)

def rustDispatcher() -> str:

    lines = [
        "/// Evaluate one generated FOIS overlap metric element.",
        "/// # Arguments:",
        "/// - `left`: Left excitation operator.",
        "/// - `right`: Right excitation operator.",
        "/// - `spaces`: Core, active, and virtual orbital-space maps.",
        "/// - `gamma1`: Spin-free one-particle RDM.",
        "/// - `lambdas`: Spin-free cumulants.",
        "/// # Returns:",
        "/// - `f64`: Raw FOIS overlap metric element, or `0.0` for orthogonal class pairs.",
        "pub(crate) fn overlap_element(",
        "    left: Excitation,",
        "    right: Excitation,",
        "    spaces: &Spaces,",
        "    gamma1: &RDM1<f64>,",
        "    lambdas: &Cumulants<f64>,",
        ") -> f64 {",
        "    let lclass = excitation_class(spaces, left);",
        "    let rclass = excitation_class(spaces, right);",
        "",
        "    match (lclass, rclass) {",
    ]

    for block in BLOCKS:
        lines.append(
            "        "
            + f"(ExcitationClass::{block.left_class}, ExcitationClass::{block.right_class}) => "
            + f"{block.function}(left, right, spaces, gamma1, lambdas),"
        )

        if block.left_class != block.right_class:
            lines.append(
                "        "
                + f"(ExcitationClass::{block.right_class}, ExcitationClass::{block.left_class}) => "
                + f"{block.function}(right, left, spaces, gamma1, lambdas),"
            )

    lines.extend([
        "        _ => 0.0,",
        "    }",
        "}",
    ])

    return "\n".join(lines)

def rustHeader() -> str:
    return """// deterministic/noccmc/overlap.rs
// This file is generated by tools/wick/rust.py.
// Do not edit generated overlap kernels by hand.

use crate::deterministic::noccmc::space::{
    Excitation,
    ExcitationClass,
    Spaces,
    excitation_class,
};
use crate::noci::{
    Cumulants,
    RDM1,
};

/// Unpack one single excitation.
/// # Arguments:
/// - `ex`: Excitation expected to be a single excitation.
/// # Returns:
/// - `(usize, usize)`: Creation and annihilation orbital indices.
fn single(ex: Excitation) -> (usize, usize) {
    match ex {
        Excitation::Single { p, q } => (p, q),
        _ => panic!("expected single excitation"),
    }
}

/// Unpack one double excitation.
/// # Arguments:
/// - `ex`: Excitation expected to be a double excitation.
/// # Returns:
/// - `(usize, usize, usize, usize)`: Two creation and two annihilation orbital indices.
fn double(ex: Excitation) -> (usize, usize, usize, usize) {
    match ex {
        Excitation::Double { p, q, r, s } => (p, q, r, s),
        _ => panic!("expected double excitation"),
    }
}

/// Evaluate a Kronecker delta.
/// # Arguments:
/// - `p`: Left orbital index.
/// - `q`: Right orbital index.
/// # Returns:
/// - `f64`: `1.0` if the indices are equal, otherwise `0.0`.
fn delta(p: usize, q: usize) -> f64 {
    if p == q {
        1.0
    } else {
        0.0
    }
}

/// Evaluate an active-space hole density.
/// # Arguments:
/// - `gamma1`: Spin-free one-particle RDM.
/// - `p`: Upper orbital index.
/// - `q`: Lower orbital index.
/// # Returns:
/// - `f64`: `Theta^p_q = 2 delta^p_q - Gamma^p_q`.
fn theta(gamma1: &RDM1<f64>, p: usize, q: usize) -> f64 {
    2.0 * delta(p, q) - gamma1.data[p * gamma1.n + q]
}

/// Convert a global orbital index to an active-space index.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `p`: Global orbital index.
/// # Returns:
/// - `usize`: Active-space index corresponding to `p`.
fn active(spaces: &Spaces, p: usize) -> usize {
    spaces.active_map[p].expect("expected active orbital index")
}
"""

def rustModule(debugBlock: str | None = None) -> str:
    parts = [
        rustHeader(),
        rustDispatcher(),
    ]

    parts.extend(
        rustFunction(
            block,
            debug = block.block == debugBlock,
        )
        for block in BLOCKS
    )

    return "\n\n".join(parts) + "\n"

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug-block",
        choices = tuple(block.block for block in BLOCKS),
        default = None,
        help = "emit term-level debug printing for one generated overlap block",
    )

    args = parser.parse_args()

    print(
        rustModule(
            debugBlock = args.debug_block,
        ),
        end = "",
    )

if __name__ == "__main__":
    main()
