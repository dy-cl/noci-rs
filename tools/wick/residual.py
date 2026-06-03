from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from itertools import product as cartesianProduct

from canonical import canonicaliseForOutput
from core import (
    Expr,
    Group,
    Idx,
    Product,
    Ref,
    Space,
    Tensor,
    Wick,
    add,
    daggerTau1,
    daggerTau2,
    groupE1,
    groupE2,
    mul,
    scale,
    tensor,
)
from latex import latexEquation, latexIndexTuple

@dataclass(frozen = True)
class ExcitationSpec:
    """One raw GNOCCSD excitation type.

    Notation:
        tau_mu = E^{creators}_{annihilators}

    Examples:
        CToA represents E^u_i.
        CAToAV represents E^{va}_{iu}.
    """
    name: str
    creators: tuple[Idx, ...]
    annihilators: tuple[Idx, ...]

def C(name: str) -> Idx:
    """Build one core orbital index.

    Notation:
        i,j,k,l in C

    Examples:
        C("i") represents core orbital i.
    """
    return Idx(name, Space.CORE)

def A(name: str) -> Idx:
    """Build one active orbital index.

    Notation:
        t,u,v,w in A

    Examples:
        A("u") represents active orbital u.
    """
    return Idx(name, Space.ACTIVE)

def V(name: str) -> Idx:
    """Build one virtual orbital index.

    Notation:
        a,b,c,d in V

    Examples:
        V("a") represents virtual orbital a.
    """
    return Idx(name, Space.VIRTUAL)

EXCITATIONS = {
    "CToA": ExcitationSpec("CToA", (A("u"),), (C("i"),)),
    "CToV": ExcitationSpec("CToV", (V("a"),), (C("i"),)),
    "AToA": ExcitationSpec("AToA", (A("v"),), (A("u"),)),
    "AToV": ExcitationSpec("AToV", (V("a"),), (A("u"),)),
    "CCToAA": ExcitationSpec("CCToAA", (A("u"), A("v")), (C("i"), C("j"))),
    "CCToAV": ExcitationSpec("CCToAV", (A("u"), V("a")), (C("i"), C("j"))),
    "CAToAA": ExcitationSpec("CAToAA", (A("v"), A("w")), (C("i"), A("u"))),
    "CAToAV": ExcitationSpec("CAToAV", (A("v"), V("a")), (C("i"), A("u"))),
    "CAToVA": ExcitationSpec("CAToVA", (V("a"), A("v")), (C("i"), A("u"))),
    "CAToVV": ExcitationSpec("CAToVV", (V("a"), V("b")), (C("i"), A("u"))),
    "AAToAA": ExcitationSpec("AAToAA", (A("t"), A("v")), (A("u"), A("w"))),
    "AAToAV": ExcitationSpec("AAToAV", (A("v"), V("a")), (A("t"), A("u"))),
    "AAToVV": ExcitationSpec("AAToVV", (V("a"), V("b")), (A("t"), A("u"))),
}

def availableExcitations() -> tuple[str, ...]:
    """Return supported raw residual excitation types.

    Notation:
        CToA, CToV, ...

    Examples:
        availableExcitations() contains AAToAA.
    """
    return tuple(EXCITATIONS.keys())

def hidx(space: Space, slot: int) -> Idx:
    """Build one Hamiltonian dummy index.

    Notation:
        Hamiltonian summation indices are distinct from residual free indices.

    Examples:
        hidx(Space.ACTIVE, 0) returns active dummy ha0.
    """
    prefixes = {
        Space.CORE: "hc",
        Space.ACTIVE: "ha",
        Space.VIRTUAL: "hv",
    }

    return Idx(f"{prefixes[space]}{slot}", space)

def allH1Indices() -> tuple[tuple[Idx, Idx], ...]:
    """Return all one-body Hamiltonian index space assignments.

    Notation:
        p,q in C union A union V

    Examples:
        One element corresponds to p in A and q in V.
    """
    spaces = (Space.CORE, Space.ACTIVE, Space.VIRTUAL)

    return tuple(
        (
            hidx(pSpace, 0),
            hidx(qSpace, 1),
        )
        for pSpace, qSpace in cartesianProduct(spaces, repeat = 2)
    )

def allH2Indices() -> tuple[tuple[Idx, Idx, Idx, Idx], ...]:
    """Return all two-body Hamiltonian index space assignments.

    Notation:
        p,q,r,s in C union A union V

    Examples:
        One element corresponds to E^{pq}_{rs} with p,q,r,s in arbitrary spaces.
    """
    spaces = (Space.CORE, Space.ACTIVE, Space.VIRTUAL)

    return tuple(
        (
            hidx(pSpace, 0),
            hidx(qSpace, 1),
            hidx(rSpace, 2),
            hidx(sSpace, 3),
        )
        for pSpace, qSpace, rSpace, sSpace in cartesianProduct(spaces, repeat = 4)
    )

def braGroup(spec: ExcitationSpec, groupId: int) -> Group:
    """Build the daggered residual projector group.

    Notation:
        tau_mu = E^{p}_{q}
        tau_mu dagger = E^{q}_{p}

    Examples:
        CToA, tau = E^u_i, gives tau dagger = E^i_u.
    """
    if len(spec.creators) == 1:
        return daggerTau1(
            spec.creators[0],
            spec.annihilators[0],
            groupId,
        )

    if len(spec.creators) == 2:
        return daggerTau2(
            spec.creators[0],
            spec.creators[1],
            spec.annihilators[0],
            spec.annihilators[1],
            groupId,
        )

    raise ValueError(f"unsupported excitation rank {len(spec.creators)}")

def h1Coeff(p: Idx, q: Idx) -> Expr:
    """Build the one-body Hamiltonian coefficient.

    Notation:
        f^q_p E^p_q

    Examples:
        h1Coeff(p, q) returns f^q_p.
    """
    return tensor("f", (q,), (p,))

def h2Coeff(p: Idx, q: Idx, r: Idx, s: Idx) -> Expr:
    """Build the two-body Hamiltonian coefficient.

    Notation:
        g^{rs}_{pq} E^{pq}_{rs}

    Examples:
        h2Coeff(p, q, r, s) returns g^{rs}_{pq}.
    """
    return tensor("g", (r, s), (p, q))

def h1Contribution(bra: Group, wick: Wick) -> Expr:
    """Build the one-body Hamiltonian contribution to R0.

    Notation:
        sum_pq f^q_p <Phi| tau_mu^dagger {E^p_q} |Phi>

    Examples:
        For CToA this gives all f-coupled terms in R_i^u.
    """
    out: Expr = ()

    for p, q in allH1Indices():
        value = wick.eval(
            Product((
                bra,
                groupE1(p, q, 1),
            ))
        )

        if value:
            out = add(
                out,
                mul(
                    h1Coeff(p, q),
                    value,
                ),
            )

    return out

def h2Contribution(bra: Group, wick: Wick) -> Expr:
    """Build the two-body Hamiltonian contribution to R0.

    Notation:
        1/2 sum_pqrs g^{rs}_{pq} <Phi| tau_mu^dagger {E^{pq}_{rs}} |Phi>

    Examples:
        For AAToAA this gives all g-coupled terms in R_{uw}^{tv}.
    """
    out: Expr = ()

    for p, q, r, s in allH2Indices():
        value = wick.eval(
            Product((
                bra,
                groupE2(p, q, r, s, 1),
            ))
        )

        if value:
            out = add(
                out,
                scale(
                    mul(
                        h2Coeff(p, q, r, s),
                        value,
                    ),
                    Fraction(1, 2),
                ),
            )

    return out

def r0Expr(name: str) -> Expr:
    """Evaluate one zeroth-order raw residual.

    Notation:
        R_mu^(0) = <Phi| tau_mu^dagger H |Phi>_c

    Examples:
        r0Expr("CToA") evaluates R_i^u at zero cluster amplitude.
    """
    spec = EXCITATIONS[name]
    wick = Wick(Ref())

    return canonicaliseForOutput(
        add(
            h1Contribution(
                braGroup(spec, 0),
                wick,
            ),
            h2Contribution(
                braGroup(spec, 0),
                wick,
            ),
        )
    )

def rustName(name: str) -> str:
    """Convert one residual class name to a Rust function suffix.

    Notation:
        CToA -> c_to_a

    Examples:
        rustName("AAToAA") returns "aa_to_aa".
    """
    out = []

    for i, char in enumerate(name):
        if i > 0 and char.isupper():
            out.append("_")

        out.append(char.lower())

    return "".join(out).replace("_to", "_to")

def rustCoeff(coeff) -> str:
    """Emit one Rust scalar coefficient.

    Notation:
        a / b

    Examples:
        Fraction(1, 2) becomes 0.5-style Rust arithmetic.
    """
    if coeff.denominator == 1:
        return f"{coeff.numerator}.0"

    return f"({coeff.numerator}.0 / {coeff.denominator}.0)"

def rustDelta(delta) -> str:
    """Emit one Rust Kronecker delta.

    Notation:
        delta^p_q

    Examples:
        delta(i, j) emits delta(i, j).
    """
    return f"delta({delta.left.name}, {delta.right.name})"

def rustTensor(tensor: Tensor) -> str:
    """Emit one Rust tensor access for R0 residual code.

    Notation:
        Gamma1, Theta, Lambda_k, f, g

    Examples:
        f^q_p emits f[(q, p)].
    """
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

def rustTermFactors(term) -> str:
    """Emit the multiplicative factors in one Rust residual term.

    Notation:
        c * delta * tensor * ...

    Examples:
        -1/2 delta f Lambda2.
    """
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

def freeIndexNames(spec: ExcitationSpec) -> set[str]:
    """Return free residual index names.

    Notation:
        R^{creators}_{annihilators}

    Examples:
        CToA has free names u and i.
    """
    return {
        idx.name
        for idx in spec.creators + spec.annihilators
    }

def termIndices(term) -> tuple[Idx, ...]:
    """Return all orbital indices used by one symbolic term.

    Notation:
        indices appearing in deltas and tensors

    Examples:
        A term with Lambda2^{ha0 u}_{v ha1} returns ha0,u,v,ha1 plus others.
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
    """Return non-free dummy indices needed as Rust summation loops.

    Notation:
        summed Hamiltonian indices

    Examples:
        ha0, ha1, hc2, hv0.
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
    """Emit one Rust loop over an orbital space.

    Notation:
        sum_{p in C/A/V}

    Examples:
        ha0 loops over spaces.active.
    """
    if idx.space == Space.CORE:
        return f"{indent}for &{idx.name} in spaces.core.iter() {{"

    if idx.space == Space.ACTIVE:
        return f"{indent}for &{idx.name} in spaces.active.iter() {{"

    if idx.space == Space.VIRTUAL:
        return f"{indent}for &{idx.name} in spaces.virtuals.iter() {{"

    raise ValueError(f"unsupported index space {idx.space}")

def rustTermAccumulation(term, spec: ExcitationSpec) -> list[str]:
    """Emit Rust loops and accumulation for one residual term.

    Notation:
        out += sum_dummy term

    Examples:
        A term containing ha0 and ha1 gets two nested loops.
    """
    dummies = dummyIndices(term, spec)
    lines = []
    indent = "    "

    for idx in dummies:
        lines.append(rustLoopOpen(idx, indent))
        indent += "    "

    lines.append(f"{indent}out += {rustTermFactors(term)};")

    for _ in dummies:
        indent = indent[:-4]
        lines.append(f"{indent}}}")

    return lines

def rustUnpack(spec: ExcitationSpec) -> str:
    """Emit Rust unpacking for one residual excitation.

    Notation:
        Excitation::Single or Excitation::Double

    Examples:
        CToA emits let (u, i) = single(ex).
    """
    names = ", ".join(idx.name for idx in spec.creators + spec.annihilators)

    if len(spec.creators) == 1:
        return f"    let ({names}) = single(ex);"

    if len(spec.creators) == 2:
        return f"    let ({names}) = double(ex);"

    raise ValueError(f"unsupported excitation rank {len(spec.creators)}")

def rustResidualFunction(name: str) -> str:
    """Emit one generated Rust R0 residual kernel.

    Notation:
        R_mu^(0)

    Examples:
        r0_c_to_a evaluates R^u_i at zero cluster amplitudes.
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
        f"fn r0_{rustName(name)}(",
        "    ex: Excitation,",
        "    ao: &AoData,",
        "    f: &Array2<f64>,",
        "    spaces: &Spaces,",
        "    gamma1: &RDM1<f64>,",
        "    lambdas: &Cumulants<f64>,",
        ") -> f64 {",
        rustUnpack(spec),
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
    """Emit the Rust R0 residual dispatcher.

    Notation:
        class(ex) -> generated kernel

    Examples:
        CToA dispatches to r0_c_to_a.
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
        lines.append(
            f"        ExcitationClass::{name} => "
            f"r0_{rustName(name)}(ex, ao, f, spaces, gamma1, lambdas),"
        )

    lines.extend([
        "    }",
        "}",
    ])

    return "\n".join(lines)

def rustResidualBuilder() -> str:
    """Emit the Rust vector builder for direct R0.

    Notation:
        R0_mu = <Phi|tau_mu^dagger H|Phi>_c

    Examples:
        r0 returns one value per raw excitation.
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

def rustHeader() -> str:
    """Emit the generated Rust residual module header.

    Notation:
        residual.rs

    Examples:
        Includes AoData, ndarray, space types, cumulants, and fock.
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

def rustModule() -> str:
    """Emit the complete generated Rust R0 residual module.

    Notation:
        src/deterministic/noccmc/residual.rs

    Examples:
        python tools/wick/residual.py --emit rust > src/deterministic/noccmc/residual.rs
    """
    parts = [
        rustHeader(),
        rustResidualBuilder(),
        rustResidualDispatcher(),
    ]

    parts.extend(
        rustResidualFunction(name)
        for name in availableExcitations()
    )

    return "\n\n".join(parts) + "\n"

def residualLatexName(spec: ExcitationSpec) -> str:
    """Return the printed residual label.

    Notation:
        R^{creators}_{annihilators,(0)}

    Examples:
        CToA gives R^{u}_{i,(0)}.
    """
    upper = latexIndexTuple(spec.creators)
    lower = latexIndexTuple(spec.annihilators)

    return rf"R^{{{upper}}}_{{{lower},(0)}}"

def emitResidual(name: str, emit: str) -> str:
    """Emit one residual expression.

    Notation:
        --class CToA emits R_i^u

    Examples:
        emitResidual("AAToAA", "latex") emits the active-active double R0.
    """
    expr = r0Expr(name)

    if emit == "latex":
        return latexEquation(
            residualLatexName(EXCITATIONS[name]),
            expr,
        )

    if emit == "expr":
        return repr(expr)

    raise ValueError(f"unknown emit mode {emit}")

def emitResiduals(name: str, emit: str) -> str:
    """Emit one residual expression or all residual expressions.

    Notation:
        --class all emits every raw GNOCCSD residual type.

    Examples:
        --class CToA emits only the C -> A residual.
    """
    if name == "all":
        return "\n\n".join(
            emitResidual(excitation, emit)
            for excitation in availableExcitations()
        )

    return emitResidual(name, emit)
def main() -> None:
    """Run the residual generator.

    Notation:
        python tools/wick/residual.py --class CToA --emit latex

    Examples:
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
        "--line-width",
        type = int,
        default = 120,
        help = "maximum LaTeX line width before wrapping; use 0 to disable wrapping",
    )

    args = parser.parse_args()
    lineWidth = None if args.line_width <= 0 else args.line_width

    if args.emit == "rust":
        print(rustModule(), end = "")
        return

    print(
        emitResiduals(
            args.excitation,
            args.emit,
            lineWidth = lineWidth,
        )
    )

if __name__ == "__main__":
    main()
