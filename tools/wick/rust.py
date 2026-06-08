from __future__ import annotations

import argparse

from core import Delta, Expr, Idx, Space, Tensor, Term
from equations import overlapExpr, outputExpr, r0Expr
from specs import EXCITATIONS, OVERLAP_BLOCKS, ExcitationSpec, OverlapBlockSpec, availableExcitations, overlapBlock

def rustFunctionDoc(block: OverlapBlockSpec) -> list[str]:
    return [
        f"/// Evaluate generated Appendix-C overlap block `{block.name}`.",
        "/// # Arguments:",
        "/// - `left`: Left excitation operator, daggered in the metric element.",
        "/// - `right`: Right excitation operator.",
        "/// - `spaces`: Core, active, and virtual orbital-space maps.",
        "/// - `gamma1`: Spin-free one-particle RDM.",
        "/// - `lambdas`: Spin-free active-space cumulants.",
        "/// # Returns:",
        "/// - `f64`: Raw FOIS overlap metric element.",
    ]

def blockByName(name: str) -> OverlapBlockSpec:
    return overlapBlock(name)

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

    if tensor.name == "f":
        q = tensor.upper[0].name
        p = tensor.lower[0].name
        return f"f[({q}, {p})]"

    if tensor.name == "g":
        r = tensor.upper[0].name
        s = tensor.upper[1].name
        p = tensor.lower[0].name
        q = tensor.lower[1].name
        return f"ao.eri_coul[({r}, {s}, {p}, {q})]"

    raise ValueError(f"unknown tensor {tensor.name}")

def rustTerm(term: Term) -> str:
    factors = []

    factors.extend(
        rustDelta(delta)
        for delta in term.deltas
    )

    factors.extend(
        rustTensor(tensor)
        for tensor in term.tensors
    )

    body = " * ".join(factors) if factors else "1.0"

    if term.coeff == 1:
        return body

    if term.coeff == -1:
        return f"-({body})"

    if term.coeff < 0:
        return f"-({rustCoeff(-term.coeff)} * {body})"

    return f"{rustCoeff(term.coeff)} * {body}"

def rustOverlapUnpack(names: tuple[str, ...], side: str) -> str:
    lhs = ", ".join(names)

    if len(names) == 2:
        return f"    let ({lhs}) = single({side});"

    if len(names) == 4:
        return f"    let ({lhs}) = double({side});"

    raise ValueError(f"unsupported unpack size {len(names)}")

def rustFunction(block: OverlapBlockSpec, debug: bool = False) -> str:
    expr = list(outputExpr(block.name))
    left_unpack, right_unpack = block.unpack

    lines = rustFunctionDoc(block) + [
        "#[allow(unused_variables)]",
        f"pub(crate) fn {block.rustName}(",
        "    left: Excitation,",
        "    right: Excitation,",
        "    spaces: &Spaces,",
        "    gamma1: &RDM1<f64>,",
        "    lambdas: &Cumulants<f64>,",
        ") -> f64 {",
        rustOverlapUnpack(left_unpack, "left"),
        rustOverlapUnpack(right_unpack, "right"),
    ]

    if not debug:
        if not expr:
            lines.extend([
                "    0.0",
                "}",
            ])

            return "\n".join(lines)

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
        f'        eprintln!("DEBUG {block.name} {{:?}}", left);',
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

def rustFunctionDebugTerms(block: OverlapBlockSpec) -> str:
    expr = overlapExpr(block.name)
    left_unpack, right_unpack = block.unpack

    lines = rustFunctionDoc(block) + [
        "#[allow(unused_variables)]",
        f"pub(crate) fn {block.rustName}(",
        "    left: Excitation,",
        "    right: Excitation,",
        "    spaces: &Spaces,",
        "    gamma1: &RDM1<f64>,",
        "    lambdas: &Cumulants<f64>,",
        ") -> f64 {",
        rustOverlapUnpack(left_unpack, "left"),
        rustOverlapUnpack(right_unpack, "right"),
    ]

    for i, term in enumerate(expr):
        lines.append(f"    let term_{i} = {rustTerm(term)};")

    terms = " + ".join(f"term_{i}" for i in range(len(expr))) or "0.0"
    lines.append(f"    let out = {terms};")

    lines.extend([
        f'    if left == right && matches!(excitation_class(spaces, left), ExcitationClass::{block.left}) {{',
        f'        eprintln!("DEBUG {block.name} {{:?}}", left);',
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

    for block in OVERLAP_BLOCKS:
        lines.append(
            "        "
            + f"(ExcitationClass::{block.left}, ExcitationClass::{block.right}) => "
            + f"{block.rustName}(left, right, spaces, gamma1, lambdas),"
        )

        if block.left != block.right:
            lines.append(
                "        "
                + f"(ExcitationClass::{block.right}, ExcitationClass::{block.left}) => "
                + f"{block.rustName}(right, left, spaces, gamma1, lambdas),"
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

def rustOverlapFunction(name: str, debug: bool = False) -> str:
    return rustFunction(
        overlapBlock(name),
        debug = debug,
    )

def rustModule(debugBlock: str | None = None) -> str:
    parts = [
        rustHeader(),
        rustDispatcher(),
    ]

    parts.extend(
        rustFunction(
            block,
            debug = block.name == debugBlock,
        )
        for block in OVERLAP_BLOCKS
    )

    return "\n\n".join(parts) + "\n"

def freeIndexNames(spec: ExcitationSpec) -> set[str]:
    return {
        idx.name
        for idx in spec.creators + spec.annihilators
    }

def termIndices(term) -> tuple[Idx, ...]:
    out = []

    for delta in term.deltas:
        out.append(delta.left)
        out.append(delta.right)

    for tensor in term.tensors:
        out.extend(tensor.upper)
        out.extend(tensor.lower)

    return tuple(out)

def dummyIndices(term, spec: ExcitationSpec) -> tuple[Idx, ...]:
    free = freeIndexNames(spec)
    seen = set()
    out = []

    for idx in termIndices(term):
        if idx.name in free or idx.name in seen:
            continue

        seen.add(idx.name)
        out.append(idx)

    return tuple(out)

def rustLoopOpen(idx: Idx, indent: str) -> str:
    if idx.space == Space.CORE:
        return f"{indent}for &{idx.name} in spaces.core.iter() {{"

    if idx.space == Space.ACTIVE:
        return f"{indent}for &{idx.name} in spaces.active.iter() {{"

    if idx.space == Space.VIRTUAL:
        return f"{indent}for &{idx.name} in spaces.virtuals.iter() {{"

    raise ValueError(f"unsupported index space {idx.space}")

def rustResidualTermFactors(term) -> str:
    factors = []

    factors.extend(
        rustDelta(delta)
        for delta in term.deltas
    )

    factors.extend(
        rustTensor(tensor)
        for tensor in term.tensors
    )

    body = " * ".join(factors) if factors else "1.0"

    if term.coeff == 1:
        return body

    if term.coeff == -1:
        return f"-({body})"

    return f"{rustCoeff(term.coeff)} * {body}"

def rustTermAccumulation(term, spec: ExcitationSpec) -> list[str]:
    dummies = dummyIndices(term, spec)
    lines = []
    indent = "    "

    for idx in dummies:
        lines.append(rustLoopOpen(idx, indent))
        indent += "    "

    lines.append(f"{indent}out += {rustResidualTermFactors(term)};")

    for _ in dummies:
        indent = indent[:-4]
        lines.append(f"{indent}}}")

    return lines

def rustResidualUnpack(spec: ExcitationSpec) -> str:
    names = ", ".join(idx.name for idx in spec.creators + spec.annihilators)

    if len(spec.creators) == 1:
        return f"    let ({names}) = single(ex);"

    if len(spec.creators) == 2:
        return f"    let ({names}) = double(ex);"

    raise ValueError(f"unsupported excitation rank {len(spec.creators)}")

def rustResidualFunction(name: str) -> str:
    spec = EXCITATIONS[name]
    expr = r0Expr(name)

    lines = [
        f"/// Evaluate generated zeroth-order residual `{name}`.",
        "/// # Arguments:",
        "/// - `ex`: Raw excitation operator.",
        "/// - `ao`: Integrals in the NOCI natural-orbital basis.",
        "/// - `f`: Spin-free Fock matrix in the NOCI natural-orbital basis.",
        "/// - `spaces`: Core, active, and virtual orbital-space maps.",
        "/// - `gamma1`: Spin-free one-particle RDM.",
        "/// - `lambdas`: Spin-free active-space cumulants.",
        "/// # Returns:",
        "/// - `f64`: Direct zeroth-order residual element.",
        "#[allow(unused_variables)]",
        f"fn r0_{spec.rustName}(",
        "    ex: Excitation,",
        "    ao: &AoData,",
        "    f: &Array2<f64>,",
        "    spaces: &Spaces,",
        "    gamma1: &RDM1<f64>,",
        "    lambdas: &Cumulants<f64>,",
        ") -> f64 {",
        rustResidualUnpack(spec),
        "    let mut out = 0.0;",
    ]

    for term in expr:
        lines.extend(rustTermAccumulation(term, spec))

    lines.extend([
        "    out",
        "}",
    ])

    return "\n".join(lines)

def rustResidualDispatcher() -> str:
    lines = [
        "/// Evaluate one generated zeroth-order residual element.",
        "/// # Arguments:",
        "/// - `ex`: Raw excitation operator.",
        "/// - `ao`: Integrals in the NOCI natural-orbital basis.",
        "/// - `f`: Spin-free Fock matrix in the NOCI natural-orbital basis.",
        "/// - `spaces`: Core, active, and virtual orbital-space maps.",
        "/// - `gamma1`: Spin-free one-particle RDM.",
        "/// - `lambdas`: Spin-free active-space cumulants.",
        "/// # Returns:",
        "/// - `f64`: Direct zeroth-order residual element.",
        "pub(crate) fn r0e(",
        "    ex: Excitation,",
        "    ao: &AoData,",
        "    f: &Array2<f64>,",
        "    spaces: &Spaces,",
        "    gamma1: &RDM1<f64>,",
        "    lambdas: &Cumulants<f64>,",
        ") -> f64 {",
        "    match excitation_class(spaces, ex) {",
    ]

    for name in availableExcitations():
        spec = EXCITATIONS[name]
        lines.append(
            f"        ExcitationClass::{name} => "
            f"r0_{spec.rustName}(ex, ao, f, spaces, gamma1, lambdas),"
        )

    lines.extend([
        "    }",
        "}",
    ])

    return "\n".join(lines)

def rustResidualBuilder() -> str:
    return """/// Build the direct zeroth-order residual vector.
/// # Arguments:
/// - `ao`: Integrals in the NOCI natural-orbital basis.
/// - `gamma1`: Spin-free one-particle RDM.
/// - `lambdas`: Spin-free active-space cumulants.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// # Returns:
/// - `Array1<f64>`: Direct zeroth-order residual vector.
pub(crate) fn r0(
    ao: &AoData,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
    spaces: &Spaces,
    excitations: &[Excitation],
) -> Array1<f64> {
    let n = gamma1.n;
    let mut da = Array2::<f64>::zeros((n, n));
    let mut db = Array2::<f64>::zeros((n, n));

    for p in 0..n {
        for q in 0..n {
            let value = 0.5 * gamma1.data[p * n + q];
            da[(p, q)] = value;
            db[(p, q)] = value;
        }
    }

    let (f, _fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let mut out = Array1::<f64>::zeros(excitations.len());

    for (mu, &ex) in excitations.iter().enumerate() {
        out[mu] = r0e(ex, ao, &f, spaces, gamma1, lambdas);
    }

    out
}
"""

def rustResidualHeader() -> str:
    return """// deterministic/noccmc/residual.rs
// This file is generated by tools/wick/residual.py.
// Do not edit generated residual kernels by hand.

use ndarray::{Array1, Array2};

use crate::AoData;
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
use crate::scf::fock;

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

def rustResidualModule() -> str:
    parts = [
        rustResidualHeader(),
        rustResidualBuilder(),
        rustResidualDispatcher(),
    ]

    parts.extend(
        rustResidualFunction(name)
        for name in availableExcitations()
    )

    return "\n\n".join(parts) + "\n"

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug-block",
        choices = tuple(block.name for block in OVERLAP_BLOCKS),
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
