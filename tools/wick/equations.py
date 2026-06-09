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
    """
    Build the GNO operator product defining one overlap block.

    Notation:
        S_{\mu\nu} = \langle \Phi | \tau_\mu^\dagger \tau_nu | \Phi \rangle.

    Examples:
        C1 corresponds to \langle \Phi | {E^i_u} {E^v_j} | \Phi \rangle.
        C4 corresponds to \langle \Phi | {E^{iu}_{va}} {E^{xb}_{jw}} | \Phi \rangle.
    """
    
    # C1: C -> A / C -> A overlap.
    # \langle \Phi | \{E^i_u\} \{E^v_j\} | \Phi \rangle
    if name == "C1":
        i = C("i")
        j = C("j")
        u = A("u")
        v = A("v")

        return Product((
            tau1(i, u, 0),
            tau1(v, j, 1),
        ))
    
    
    # C2: A -> V / A -> V overlap.
    # \langle \Phi | \{E^t_a\} \{E^b_u\} | \Phi \rangle
    if name == "C2":
        a = V("a")
        b = V("b")
        t = A("t")
        u = A("u")

        return Product((
            tau1(t, a, 0),
            tau1(b, u, 1),
        ))
    
    # C3: A -> A / A -> A overlap.
    # \langle \Phi | \{E^u_v\} \{E^x_w\} | \Phi \rangle
    if name == "C3":
        u = A("u")
        v = A("v")
        w = A("w")
        x = A("x")

        return Product((
            tau1(u, v, 0),
            tau1(x, w, 1),
        ))
    
    # C4: CA -> AV / CA -> AV overlap.
    # \langle \Phi | \{E^{iu}_{va}\} \{E^{xb}_{jw}\} | \Phi \rangle
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
    
    # C5: CA -> VA / CA -> VA overlap.
    # \langle \Phi | \{E^{iu}_{av}\} \{E^{bx}_{jw}\} | \Phi \rangle
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
    
    # C6: CA -> VV / CA -> VV overlap.
    # \langle \Phi | \{E^{iu}_{ab}\} \{E^{cd}_{jv}\} | \Phi \rangle
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
    
    # C7: CC -> AV / CC -> AV overlap.
    # \langle \Phi | \{E^{ij}_{ua}\} \{E^{vb}_{kl}\} | \Phi \rangle
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
    
    # C8: CC -> AA / CC -> AA overlap.
    # \langle \Phi | \{E^{ij}_{uv}\} \{E^{wx}_{kl}\} | \Phi \rangle
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
    
    # C9: CA -> AA / CA -> AA overlap.
    # \langle \Phi | \{E^{iu}_{vw}\} \{E^{yz}_{jx}\} | \Phi \rangle
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
    
    # C10: AA -> AV / AA -> AV overlap.
    # \langle \Phi | \{E^{tu}_{va}\} \{E^{zb}_{xy}\} | \Phi \rangle
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
    
    # C11: AA -> VV / AA -> VV overlap.
    # \langle \Phi | \{E^{tu}_{ab}\} \{E^{cd}_{vw}\} | \Phi \rangle
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
    
    # C12: AA -> AA / AA -> AA overlap.
    # \langle \Phi | \{E^{pr}_{qs}\} \{E^{tv}_{uw}\} | \Phi \rangle
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
    
    # C13: A -> V / AA -> AV mixed overlap.
    # \langle \Phi | \{E^u_a\} \{E^{xb}_{vw}\} | \Phi \rangle
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
    
    # C14: C -> A / CA -> AA mixed overlap.
    # \langle \Phi | \{E^i_u\} \{E^{wx}_{jv}\} | \Phi \rangle
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
    
    # C15: A -> A / AA -> AA mixed overlap.
    # \langle \Phi | \{E^t_u\} \{E^{yz}_{wx}\} | \Phi \rangle
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
    
    # C16: CA -> AV / CA -> VA mixed overlap.
    # \langle \Phi | \{E^{iu}_{wa}\} \{E^{by}_{jx}\} | \Phi \rangle
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
    
    # C17: C -> V / C -> V overlap.
    # \langle \Phi | \{E^i_a\} \{E^b_j\} | \Phi \rangle
    if name == "C17":
        i = C("i")
        j = C("j")
        a = V("a")
        b = V("b")

        return Product((
            tau1(i, a, 0),
            tau1(b, j, 1),
        ))
    
    # C18: C -> V / CA -> AV mixed overlap.
    # \langle \Phi | \{E^i_a\} \{E^{xb}_{jw}\} | \Phi \rangle
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
    
    # C19: C -> V / CA -> VA mixed overlap.
    # \langle \Phi | \{E^i_a\} \{E^{bx}_{jw}\} | \Phi \rangle
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
    """
    Evaluate one overlap block by Wick algebra.

    Notation:
        S_{\mu\nu} = \langle \Phi| \tau_\mu^dagger \tau_\nu | \Phi \rangle

    Examples:
        overlapExpr("C1") evaluates \langle \Phi | {E^i_u} {E^v_j} | \Phi \rangle.
    """
    return Wick(Ref()).eval(
        blockProduct(name)
    )

def outputExpr(name: str):
    """
    Evaluate and canonicalise one overlap block for emission.

    Notation:
        outputExpr(Ck) = canonicaliseForOutput(overlapExpr(Ck))

    Examples:
        outputExpr("C4") gives the canonical emitted expression for block C4.
    """
    return canonicaliseForOutput(
        overlapExpr(name)
    )

def hidx(space: Space, slot: int) -> Idx:
    """
    Build one Hamiltonian dummy index. 

    Notation:
        hcN \in C
        haN \in A
        hvN \in V

    Examples:
        hidx(Space.CORE, 0) returns hc0.
        hidx(Space.ACTIVE, 2) returns ha2.
    """
    prefixes = {
        Space.CORE: "hc",
        Space.ACTIVE: "ha",
        Space.VIRTUAL: "hv",
    }

    return Idx(f"{prefixes[space]}{slot}", space)

def allH1Indices() -> tuple[tuple[Idx, Idx], ...]:
    """
    Return supported one-body Hamiltonian index space assignments. This enumerates one-body 
    Hamiltonian operators whose index spaces match one of the available one-body excitation classes.

    Notation:
        H_1 = \sum_{p \in A \cup V} \sum_{q \in C \cup A} f^q_p \{E^p_q\}

    Examples:
        If C -> A is an available excitation class, this includes p \in A, q \in C
        as one possible Hamiltonian one-body index assignment. If A -> V is an available excitation class, 
        this includes p \in V, q \in A.
    """
    return tuple(
        (
            hidx(spec.creators[0].space, 0),
            hidx(spec.annihilators[0].space, 1),
        )
        for spec in EXCITATIONS.values()
        if len(spec.creators) == 1
    )

def allH2Indices() -> tuple[tuple[Idx, Idx, Idx, Idx], ...]:
    """
    Return supported two-body Hamiltonian index space assignments. This enumerates two-body 
    Hamiltonian insertions whose index spaces match the available two-body excitation classes.
    
    Notation:
        H_2 = \frac{1}{2} \sum_{p,q \in A \cup V} \sum_{r,s \in C \cup A} g^{rs}_{pq} \{E^{pq}_{rs}\}

    Examples:
        If CA -> AV is an available excitation class, this includes p \in A, q \in V, r \in C, s \in A
        as one possible Hamiltonian two-body index assignment. If AA -> VV is an available excitation class, 
        this includes p,q \in V, r,s \in A.
    """
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
    """
    Build the daggered residual projector group. Converts an excitation specfication 
    into the adjoint GNO operator.

    Notation:
        If \tau_\mu = \{E^p_q\} then \tau_\mu^\dagger = \{E^q_p\}
        If \tau_\mu = \{E^{pq}_{rs}\} then \tau_\mu^\dagger = \{E^{rs}_{pq}\}

    Examples:
        For a one-body C -> A excitation, \tau_\mu = \{E^u_i\}
        this returns \tau_\mu^\dagger = \{E^i_u\}.

        For a two-body CA -> AV excitation, \tau_\mu = \{E^{va}_{iu}\}
        this returns \tau_\mu^\dagger = \{E^{iu}_{va}\}.
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
    """
    Build the one-body Hamiltonian coefficient. Returns only the scalar coefficient attatched to \{E^p_q\}.
    The Wick expectation value is constructed seperately in h1Contribution.

    Notation:
        f^q_p \{E^p_q\} (function returns f^q_p)

    Examples:
        h1Coeff(p, q) returns f^q_p.
    """
    return tensor("f", (q,), (p,))

def h2Coeff(p: Idx, q: Idx, r: Idx, s: Idx) -> Expr:
    """
    Build the two-body Hamiltonian coefficient. This returns only the scalar coefficient attached to \{E^{pq}_{rs}\}.
    The Hamiltonian prefactor \frac{1}{2} is applied in h2Contribution.

    Notation:
        \frac{1}{2} g^{rs}_{pq} \{E^{pq}_{rs}\} (function returns g^{rs}_{pq})

    Examples:
        h2Coeff(p, q, r, s) returns the tensor g^{rs}_{pq}.
    """
    return tensor("g", (r, s), (p, q))

def h1Contribution(bra: Group, wick: Wick) -> Expr:
    """
    Build the one-body Hamiltonian contribution to the zeroth-order residual. This evaluates all 
    supported one-body Hamiltonian insertions against the given daggered bra projector.

    Notation:
        R_\mu^{(0,1)} 
        = \sum_{p \in A \cup V} \sum_{q \in C \cup A} f^q_p
        \langle \Phi | \tau_\mu^\dagger \{E^p_q\} | \Phi \rangle.
    
    Examples:
        For a bra group \tau_\mu^\dagger = \{E^i_u\}, this evaluates all supported
        one-body Hamiltonian operators \{E^p_q\} and keeps the non-zero Wick
        contractions.
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
    """
    Build the two-body Hamiltonian contribution to the zeroth-order residual. This evaluates all 
    supported two-body Hamiltonian insertions against the given daggered bra projector.

    Notation:
        R_\mu^{(0,2)} 
        = \frac{1}{2} \sum_{p,q \in A \cup V}\sum_{r,s \in C \cup A} g^{rs}_{pq}
        \langle \Phi | \tau_\mu^\dagger \{E^{pq}_{rs}\} | \Phi \rangle

    Examples:
        For a bra group \tau_\mu^\dagger = \{E^{iu}_{va}\}, this tests all
        supported two-body Hamiltonian operators \{E^{pq}_{rs}\} and keeps
        the non-zero Wick contractions.
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
    """
    Evaluate one zeroth-order raw residual.

    Notation:
        R_\mu^{(0)}
        = \langle \Phi | \tau_\mu^\dagger H | \Phi \rangle

        H = \sum_{p \in A \cup V} \sum_{q \in C \cup A} f^q_p \{E^p_q\}
        + \frac{1}{2} \sum_{p,q \in A \cup V} \sum_{r,s \in C \cup A} g^{rs}_{pq} \{E^{pq}_{rs}\}

    Examples:
        r0Expr("CtoA") evaluates the zeroth-order residual for the C -> A
        excitation class, if "CtoA" is present in EXCITATIONS.
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
