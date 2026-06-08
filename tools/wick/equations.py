from __future__ import annotations

from fractions import Fraction

from canonical import canonicaliseForOutput
from core import (
    Expr,
    Group,
    Idx,
    Product,
    Ref,
    Space,
    Wick,
    add,
    daggerTau1,
    daggerTau2,
    groupE1,
    groupE2,
    mul,
    scale,
    tau1,
    tensor,
)
from specs import A, C, EXCITATIONS, ExcitationSpec, V

def blockProduct(name: str) -> Product:
    """Build the GNO operator product defining one overlap block."""
    if name == "C1":
        i = C("i")
        j = C("j")
        u = A("u")
        v = A("v")

        return Product((
            tau1(i, u, 0),
            tau1(v, j, 1),
        ))

    if name == "C2":
        a = V("a")
        b = V("b")
        t = A("t")
        u = A("u")

        return Product((
            tau1(t, a, 0),
            tau1(b, u, 1),
        ))

    if name == "C3":
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            tau1(u, v, 0),
            tau1(x, w, 1),
        ))

    if name == "C4":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            groupE2(i, u, v, a, 0),
            groupE2(x, b, j, w, 1),
        ))

    if name == "C5":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            groupE2(i, u, a, v, 0),
            groupE2(b, x, j, w, 1),
        ))

    if name == "C6":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        c = V("c")
        d = V("d")
        u = A("u")
        v = A("v")

        return Product((
            groupE2(i, u, a, b, 0),
            groupE2(c, d, j, v, 1),
        ))

    if name == "C7":
        i = C("i")
        j = C("j")
        k = C("k")
        l = C("l")
        a = V("a")
        b = V("b")
        u = A("u")
        v = A("v")

        return Product((
            groupE2(i, j, u, a, 0),
            groupE2(v, b, k, l, 1),
        ))

    if name == "C8":
        i = C("i")
        j = C("j")
        k = C("k")
        l = C("l")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            groupE2(i, j, u, v, 0),
            groupE2(w, x, k, l, 1),
        ))

    if name == "C9":
        i = C("i")
        j = C("j")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")
        y = A("y")
        z = A("z")

        return Product((
            groupE2(i, u, v, w, 0),
            groupE2(y, z, j, x, 1),
        ))

    if name == "C10":
        a = V("a")
        b = V("b")
        t = A("t")
        u = A("u")
        v = A("v")
        x = A("x")
        y = A("y")
        z = A("z")

        return Product((
            groupE2(t, u, v, a, 0),
            groupE2(z, b, x, y, 1),
        ))

    if name == "C11":
        a = V("a")
        b = V("b")
        c = V("c")
        d = V("d")
        t = A("t")
        u = A("u")
        v = A("v")
        w = A("w")

        return Product((
            groupE2(t, u, a, b, 0),
            groupE2(c, d, v, w, 1),
        ))

    if name == "C12":
        p = A("p")
        q = A("q")
        r = A("r")
        s = A("s")
        t = A("t")
        u = A("u")
        v = A("v")
        w = A("w")

        return Product((
            groupE2(p, r, q, s, 0),
            groupE2(t, v, u, w, 1),
        ))

    if name == "C13":
        a = V("a")
        b = V("b")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            tau1(u, a, 0),
            groupE2(x, b, v, w, 1),
        ))

    if name == "C14":
        i = C("i")
        j = C("j")
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            tau1(i, u, 0),
            groupE2(w, x, j, v, 1),
        ))

    if name == "C15":
        t = A("t")
        u = A("u")
        w = A("w")
        x = A("x")
        y = A("y")
        z = A("z")

        return Product((
            tau1(t, u, 0),
            groupE2(y, z, w, x, 1),
        ))

    if name == "C16":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        u = A("u")
        w = A("w")
        x = A("x")
        y = A("y")

        return Product((
            groupE2(i, u, w, a, 0),
            groupE2(b, y, j, x, 1),
        ))

    if name == "C17":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")

        return Product((
            tau1(i, a, 0),
            tau1(b, j, 1),
        ))

    if name == "C18":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        w = A("w")
        x = A("x")

        return Product((
            tau1(i, a, 0),
            groupE2(x, b, j, w, 1),
        ))

    if name == "C19":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")
        w = A("w")
        x = A("x")

        return Product((
            tau1(i, a, 0),
            groupE2(b, x, j, w, 1),
        ))

    raise ValueError(f"unknown overlap block {name}")

def overlapExpr(name: str):
    """Evaluate one overlap block by Wick algebra."""
    return Wick(Ref()).eval(
        blockProduct(name)
    )

def outputExpr(name: str):
    """Evaluate and canonicalise one overlap block for emission."""
    return canonicaliseForOutput(
        overlapExpr(name)
    )

def hidx(space: Space, slot: int) -> Idx:
    """Build one Hamiltonian dummy index."""
    prefixes = {
        Space.CORE: "hc",
        Space.ACTIVE: "ha",
        Space.VIRTUAL: "hv",
    }

    return Idx(f"{prefixes[space]}{slot}", space)

def allH1Indices() -> tuple[tuple[Idx, Idx], ...]:
    """Return supported one-body Hamiltonian index space assignments."""
    return tuple(
        (
            hidx(spec.creators[0].space, 0),
            hidx(spec.annihilators[0].space, 1),
        )
        for spec in EXCITATIONS.values()
        if len(spec.creators) == 1
    )

def allH2Indices() -> tuple[tuple[Idx, Idx, Idx, Idx], ...]:
    """Return supported two-body Hamiltonian index space assignments."""
    return tuple(
        (
            hidx(spec.creators[0].space, 0),
            hidx(spec.creators[1].space, 1),
            hidx(spec.annihilators[0].space, 2),
            hidx(spec.annihilators[1].space, 3),
        )
        for spec in EXCITATIONS.values()
        if len(spec.creators) == 2
    )

def braGroup(spec: ExcitationSpec, groupId: int) -> Group:
    """Build the daggered residual projector group."""
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
    """Build the one-body Hamiltonian coefficient."""
    return tensor("f", (q,), (p,))

def h2Coeff(p: Idx, q: Idx, r: Idx, s: Idx) -> Expr:
    """Build the two-body Hamiltonian coefficient."""
    return tensor("g", (r, s), (p, q))

def h1Contribution(bra: Group, wick: Wick) -> Expr:
    """Build the one-body Hamiltonian contribution to R0."""
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
    """Build the two-body Hamiltonian contribution to R0."""
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
    """Evaluate one zeroth-order raw residual."""
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
