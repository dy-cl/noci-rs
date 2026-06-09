from __future__ import annotations

import argparse

from core import Delta, Expr, Idx, Space, Tensor, Term
from equations import overlapExpr, outputExpr, r0Expr
from specs import EXCITATIONS, OVERLAP_BLOCKS, ExcitationSpec, OverlapBlockSpec, availableExcitations, overlapBlock

def rustFunctionDoc(block: OverlapBlockSpec) -> list[str]:
    """
    Emit the Rust doc comment for one overlap block function.

    Notation:
        S_{\mu\nu} = \langle \Phi | \tau_\mu^\dagger \tau_\nu | \Phi \rangle

    Examples:
        rustFunctionDoc(blockByName("C4")) emits the Rust documentation for
        the C4 overlap block.
    """
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
    """
    Return one overlap block specification. This is currently just a wrapper.
    Candidate for deletion.

    Notation:
        C_k -> OverlapBlockSpec(C_k)

    Examples:
        blockByName("C4") returns the specification for overlap block C4.
    """
    return overlapBlock(name)

def rustCoeff(coeff) -> str:
    """
    Emit one Rust scalar coefficient. Coefficients are emitted as f64. Rational 
    coefficients are emitted as floating-point divisions.

    Notation:
        a ---> a.0
        a / b ---> (a.0 / b.0)

    Examples:
        Fraction(2, 1) becomes 2.0.
        Fraction(1, 2) becomes (1.0 / 2.0).
    """
    if coeff.denominator == 1:
        return f"{coeff.numerator}.0"

    return f"({coeff.numerator}.0 / {coeff.denominator}.0)"

def rustDelta(delta: Delta) -> str:
    """
    Emit one Rust Kronecker delta call.

    Notation:
        \delta^p_q -> delta(p, q)

    Examples:
        Delta(i, j) becomes delta(i, j).
    """
    return f"delta({delta.left.name}, {delta.right.name})"

def rustTensor(tensor: Tensor) -> str:
    """
    Emit one Rust tensor access.

    Notation:
        \Gamma^p_q ---> gamma1.data[p * gamma1.n + q]
        \Theta^p_q ---> theta(gamma1, p, q)
        \Lambda^{pq}_{rs} ---> lambdas.lambda2.get(...)
        f^q_p ---> f[(q, p)]
        g^{rs}_{pq} ---> ao.eri_coul[(r, s, p, q)]

    Examples:
        Tensor("Gamma1", (u,), (v,)) becomes gamma1.data[u * gamma1.n + v].
        Tensor("Lambda2", (u, x), (v, w)) becomes a lambda2.get(...) call
        after converting global active orbital indices to active-space indices.
    """
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
    """
    Emit one Rust expression term.

    Notation:
        c \delta \Gamma \Lambda

    Examples:
        \frac{1}{2}\delta^i_j\Gamma^u_w\Theta^x_v
        becomes (1.0 / 2.0) * delta(i, j) * gamma1.data[...] * theta(...).
    """
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
    """
    Emit Rust code to unpack one overlap excitation.

    Notation:
        Single(p, q) ---> (p, q)
        Double(p, q, r, s) ---> (p, q, r, s)

    Examples:
        rustOverlapUnpack(("i", "u"), "left") gives let (i, u) = single(left);
        rustOverlapUnpack(("i", "u", "v", "a"), "right") gives let (i, u, v, a) = double(right);
    """
    lhs = ", ".join(names)

    if len(names) == 2:
        return f"    let ({lhs}) = single({side});"

    if len(names) == 4:
        return f"    let ({lhs}) = double({side});"

    raise ValueError(f"unsupported unpack size {len(names)}")

def rustFunction(block: OverlapBlockSpec, debug: bool = False) -> str:
    """
    Emit one Rust overlap block function.

    Notation:
        S_{\mu\nu} = \langle \Phi | \tau_\mu^\dagger \tau_\nu | \Phi \rangle

    Examples:
        rustFunction(blockByName("C4")) emits the generated Rust kernel for
        the C4 overlap block.
        rustFunction(blockByName("C4"), debug = True) emits the same kernel
        with term-level debug printing.
    """
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
    """
    Emit one Rust overlap block function using raw pre-canonical terms.

    Notation:

    Examples:
    """
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
    """
    Emit the Rust overlap dispatcher.
    The dispatcher classifies the left and right excitations and routes them
    to the generated overlap block function. For symmetric off-diagonal block
    pairs, the reversed class ordering is handled by swapping left and right.

    Notation:

    Examples:
        (CA -> AV, CA -> VA) dispatches to the corresponding C16-style block.
    """

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
    """
    Emit the fixed Rust header for the overlap module. This contains imports and small helper 
    functions shared by all generated overlap kernels.

    Notation:

    Examples:
        rustHeader() emits Rust definitions for single(...), double(...),
        delta(...), theta(...), and active(...).
    """
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
    """
    Emit one named overlap block Rust function.

    Notation:
        C_k -> rustFunction(OverlapBlockSpec(C_k))

    Examples:
        rustOverlapFunction("C4") emits the C4 Rust overlap function.
    """
    return rustFunction(
        overlapBlock(name),
        debug = debug,
    )

def rustModule(debugBlock: str | None = None) -> str:
    """
    Emit the complete Rust overlap module. This concatenates the fixed header, dispatcher, 
    and every generated overlap block function.

    Notation:
        overlap.rs = header + dispatcher + \sum_k function(C_k)

    Examples:
        rustModule() emits the complete production overlap module.
    """
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
    """
    Return the free index names of one excitation specification.

    Notation:
        \tau_\mu = \tau^{p_1 \cdots p_k}_{q_1 \cdots q_k}
        f = {p_1, ..., p_k, q_1, ..., q_k} (free).

    Examples:
        For C -> A with indices (u, i), this returns {"u", "i"}.
    """
    return {
        idx.name
        for idx in spec.creators + spec.annihilators
    }

def termIndices(term) -> tuple[Idx, ...]:
    """
    Return all indices appearing in one symbolic term.

    Notation:
        \delta^i_j \Gamma^u_w \Lambda^{xy}_{vz} -> (i, j, u, w, x, y, v, z)

    Examples:
    """
    out = []

    for delta in term.deltas:
        out.append(delta.left)
        out.append(delta.right)

    for tensor in term.tensors:
        out.extend(tensor.upper)
        out.extend(tensor.lower)

    return tuple(out)

def dummyIndices(term, spec: ExcitationSpec) -> tuple[Idx, ...]:
    """
    Return dummy indices that must be looped over for one residual term.

    Notation:
        dummy(term) = indices(term) \setminus free(\tau_\mu)

    Examples:
        For a residual term containing h_a0, h_c1, u, i with excitation 
        free indices u, i, then this returns h_a0 and h_c1.
    """
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
    """
    Emit one Rust loop over the orbital space of a dummy index.

    Notation:
        i \in C -> for &i in spaces.core.iter()
        u \in A -> for &u in spaces.active.iter()
        a \in V -> for &a in spaces.virtuals.iter()

    Examples:
        rustLoopOpen(Idx("hc0", Space.CORE), "    ") emits a loop over spaces.core.
    """
    if idx.space == Space.CORE:
        return f"{indent}for &{idx.name} in spaces.core.iter() {{"

    if idx.space == Space.ACTIVE:
        return f"{indent}for &{idx.name} in spaces.active.iter() {{"

    if idx.space == Space.VIRTUAL:
        return f"{indent}for &{idx.name} in spaces.virtuals.iter() {{"

    raise ValueError(f"unsupported index space {idx.space}")

def rustResidualTermFactors(term) -> str:
    """
    Emit one residual term product. Analogous to rustTerm() so may be generalised.

    Notation:
        c \delta f g \Gamma \Lambda

    Examples:
        f^q_p \Lambda^{ux}_{vw} becomes f[(q, p)] * lambdas.lambda2.get(...).
    """
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
    """
    Emit Rust accumulation code for one residual term.

    Notation:

    Examples:
        A residual term containing one active dummy index ha0 emits
        for &ha0 in spaces.active.iter() {
            out += ...
        }
    """
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
    """
    Emit Rust code to unpack one residual excitation. Potential duplication of rustOverlapUnpack().
    Candidate for deletion or generalisation.

    Notation:
        \tau^p_q ---> let (p, q) = single(ex)
        \tau^{pq}_{rs} ---> let (p, q, r, s) = double(ex)

    Examples:
        For C ---> A with creators (u,) and annihilators (i,), this emits let (u, i) = single(ex);
    """
    names = ", ".join(idx.name for idx in spec.creators + spec.annihilators)

    if len(spec.creators) == 1:
        return f"    let ({names}) = single(ex);"

    if len(spec.creators) == 2:
        return f"    let ({names}) = double(ex);"

    raise ValueError(f"unsupported excitation rank {len(spec.creators)}")

def rustResidualFunction(name: str) -> str:
    """
    Emit one Rust zeroth-order residual function.

    Notation:
        R_\mu^{(0)} = \langle \Phi | \tau_\mu^\dagger H | \Phi \rangle

    Examples:
        rustResidualFunction("CtoA") emits the generated Rust function for the C -> A residual class.
    """
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
    """
    Emit the Rust residual dispatcher.

    Notation:
        r0e(ex) = R_\mu^{(0)}

    Examples:
        ExcitationClass::CtoA dispatches to r0_ctoa(...), using the Rust name stored in its ExcitationSpec.
    """
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
    """
    Emit the fixed Rust wrapper that builds the full zeroth-order residual vector.
    Possible duplication here with other Fock functions in the code (see SCF).

    Notation:
        r_\mu^{(0)} = r0e(\tau_\mu)

    Examples:
        rustResidualBuilder() emits the Rust function pub(crate) fn r0(...)
    """
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
    """
    Emit the fixed Rust header for the residual module. Potential duplication with 
    rustHeader(). Candidate for deletion or generalisation. 

    Notation:

    Examples:
        rustResidualHeader() emits Rust definitions for single(...),
        double(...), delta(...), theta(...), and active(...).
    """
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
    """
    Emit the complete Rust residual module. Potential duplication with rustModule.
    Candidate for deletion or generalisation.

    Notation:

    Examples:
        rustResidualModule() emits the complete generated residual module.
    """
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
    """
    Run the Rust overlap module generator CLI. This command emits the generated overlap module to stdout. 
    It is mainly used when regenerating deterministic/noccmc/overlap.rs, but the residual emission 
    logic should also be moved here.

    Notation:

    Examples:
        python tools/wick/rust.py
        python tools/wick/rust.py --debug-block C4
    """
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
