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
    combine,
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

def allOrbitalSpaces() -> tuple[Space, ...]:
    """
    Return all orbital spaces used in generated Hamiltonian insertions.

    Notation:
        X in {C, A, V}

    Examples:
        allOrbitalSpaces() returns the core, active, and virtual spaces.
    """
    return (
        Space.CORE,
        Space.ACTIVE,
        Space.VIRTUAL,
    )

def allGeneralH1Indices() -> tuple[tuple[Idx, Idx], ...]:
    """
    Return all one-body Hamiltonian index space assignments.

    This is the general Hamiltonian enumeration used beyond zeroth order.
    Unlike allH1Indices(), this includes blocks such as C -> C and V -> V,
    which can contribute after a cluster excitation has acted.

    Notation:
        H_1 = \\sum_{p,q \\in C \\cup A \\cup V} f^q_p \\{E^p_q\\}

    Examples:
        Includes p in V, q in V for virtual-virtual Fock couplings.
        Includes p in C, q in C for core-core Fock couplings.
    """
    return tuple(
        (
            hidx(pSpace, 0),
            hidx(qSpace, 1),
        )
        for pSpace in allOrbitalSpaces()
        for qSpace in allOrbitalSpaces()
    )

def allGeneralH2Indices() -> tuple[tuple[Idx, Idx, Idx, Idx], ...]:
    """
    Return all two-body Hamiltonian index space assignments.

    This is the general two-body Hamiltonian enumeration used beyond
    zeroth order. It intentionally covers all ordered C/A/V assignments
    before Wick contraction and canonical simplification remove vanishing
    terms.

    Notation:
        H_2 =
        \\frac{1}{2}
        \\sum_{p,q,r,s \\in C \\cup A \\cup V}
        g^{rs}_{pq} \\{E^{pq}_{rs}\\}

    Examples:
        Includes VV -> CC, AV -> CA, AA -> AA, and all other ordered
        space assignments.
    """
    return tuple(
        (
            hidx(pSpace, 0),
            hidx(qSpace, 1),
            hidx(rSpace, 2),
            hidx(sSpace, 3),
        )
        for pSpace in allOrbitalSpaces()
        for qSpace in allOrbitalSpaces()
        for rSpace in allOrbitalSpaces()
        for sSpace in allOrbitalSpaces()
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

def ketGroup(spec: ExcitationSpec, groupId: int) -> Group:
    """
    Build a non-daggered cluster excitation group.

    Notation:
        \\tau_\\nu = \\{E^p_q\\}
        \\tau_\\nu = \\{E^{pq}_{rs}\\}

    Examples:
        For a one-body C -> A excitation, this returns \\{E^u_i\\}.
        For a two-body CA -> AV excitation, this returns \\{E^{va}_{iu}\\}.
    """
    if len(spec.creators) == 1:
        return tau1(
            spec.creators[0],
            spec.annihilators[0],
            groupId,
        )

    if len(spec.creators) == 2:
        return groupE2(
            spec.creators[0],
            spec.creators[1],
            spec.annihilators[0],
            spec.annihilators[1],
            groupId,
        )

    raise ValueError(f"unsupported excitation rank {len(spec.creators)}")

def h1Terms(groupId: int, general: bool = False) -> tuple[tuple[Expr, Group], ...]:
    """
    Build one-body Hamiltonian terms.

    Notation:
        f^q_p \\{E^p_q\\}

    Examples:
        h1Terms(1, general = False) gives the restricted zeroth-order
        Hamiltonian insertions.
        h1Terms(1, general = True) gives all one-body Hamiltonian
        insertions for amplitude-dependent residual terms.
    """
    indices = allGeneralH1Indices() if general else allH1Indices()

    return tuple(
        (
            h1Coeff(p, q),
            groupE1(p, q, groupId),
        )
        for p, q in indices
    )

def h2Terms(groupId: int, general: bool = False) -> tuple[tuple[Expr, Group], ...]:
    """
    Build two-body Hamiltonian terms.

    Notation:
        \\frac{1}{2} g^{rs}_{pq} \\{E^{pq}_{rs}\\}

    Examples:
        h2Terms(1, general = False) gives the restricted zeroth-order
        Hamiltonian insertions.
        h2Terms(1, general = True) gives all two-body Hamiltonian
        insertions for amplitude-dependent residual terms.
    """
    indices = allGeneralH2Indices() if general else allH2Indices()

    return tuple(
        (
            scale(
                h2Coeff(p, q, r, s),
                Fraction(1, 2),
            ),
            groupE2(p, q, r, s, groupId),
        )
        for p, q, r, s in indices
    )

def hTerms(groupId: int, general: bool = False) -> tuple[tuple[Expr, Group], ...]:
    """
    Build all normal-ordered Hamiltonian terms.

    Notation:
        H = H_1 + H_2

    Examples:
        hTerms(1) is suitable for R_mu^(0).
        hTerms(1, general = True) is suitable for R_mu^(1).
    """
    return h1Terms(groupId, general = general) + h2Terms(groupId, general = general)

def addSpaceBalance(
    balance: dict[Space, int],
    creators: tuple[Idx, ...],
    annihilators: tuple[Idx, ...],
) -> dict[Space, int]:
    """
    Add one operator space balance.

    Notation:
        B_X = n_create(X) - n_annihilate(X)

    Examples:
        E^u_i adds +1 to active and -1 to core.
    """
    out = dict(balance)

    for idx in creators:
        out[idx.space] = out.get(idx.space, 0) + 1

    for idx in annihilators:
        out[idx.space] = out.get(idx.space, 0) - 1

    return out

def excitationBalance(
    spec: ExcitationSpec,
    daggered: bool = False,
) -> dict[Space, int]:
    """
    Return the orbital-space balance of one excitation operator.

    Notation:
        \tau = E^{p\cdots}_{q\cdots}

    Examples:
        C -> A has active:+1 and core:-1.
        The daggered C -> A operator has core:+1 and active:-1.
    """
    if daggered:
        return addSpaceBalance(
            {},
            spec.annihilators,
            spec.creators,
        )

    return addSpaceBalance(
        {},
        spec.creators,
        spec.annihilators,
    )

def h1Balance(p: Idx, q: Idx) -> dict[Space, int]:
    """
    Return the orbital-space balance of one one-body Hamiltonian operator.

    Notation:
        E^p_q

    Examples:
        E^a_i has virtual:+1 and core:-1.
    """
    return addSpaceBalance(
        {},
        (p,),
        (q,),
    )

def h2Balance(
    p: Idx,
    q: Idx,
    r: Idx,
    s: Idx,
) -> dict[Space, int]:
    """
    Return the orbital-space balance of one two-body Hamiltonian operator.

    Notation:
        E^{pq}_{rs}

    Examples:
        E^{ab}_{ij} has virtual:+2 and core:-2.
    """
    return addSpaceBalance(
        {},
        (
            p,
            q,
        ),
        (
            r,
            s,
        ),
    )

def mergeBalances(*balances: dict[Space, int]) -> dict[Space, int]:
    """
    Add several orbital-space balances.

    Notation:
        B = B_1 + B_2 + \cdots

    Examples:
        mergeBalances(braBalance, hBalance, tBalance) gives the
        total balance of one residual Wick product.
    """
    out: dict[Space, int] = {}

    for balance in balances:
        for space, value in balance.items():
            out[space] = out.get(space, 0) + value

    return {
        space: value
        for space, value in out.items()
        if value != 0
    }

def negateBalance(balance: dict[Space, int]) -> dict[Space, int]:
    """
    Negate an orbital-space balance.

    Notation:
        -B

    Examples:
        If B has active:+1, the negated balance has active:-1.
    """
    return {
        space: -value
        for space, value in balance.items()
        if value != 0
    }

def sameBalance(
    left: dict[Space, int],
    right: dict[Space, int],
) -> bool:
    """
    Compare two orbital-space balances.

    Notation:
        B_left = B_right

    Examples:
        Missing zero-valued spaces are ignored.
    """
    spaces = set(left) | set(right)

    return all(
        left.get(space, 0) == right.get(space, 0)
        for space in spaces
    )

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

def h1TermsWithBalance(
    groupId: int,
    required: dict[Space, int],
) -> tuple[tuple[Expr, Group], ...]:
    """
    Build one-body Hamiltonian terms with a required balance.

    Notation:
        f^q_p \{E^p_q\}

    Examples:
        If required is virtual:+1 and core:-1, this keeps C -> V-like
        one-body Hamiltonian insertions.
    """
    out = []

    for p, q in allGeneralH1Indices():
        if not sameBalance(
            h1Balance(p, q),
            required,
        ):
            continue

        out.append((
            h1Coeff(p, q),
            groupE1(p, q, groupId),
        ))

    return tuple(out)

def h2TermsWithBalance(
    groupId: int,
    required: dict[Space, int],
) -> tuple[tuple[Expr, Group], ...]:
    """
    Build two-body Hamiltonian terms with a required balance.

    Notation:
        \frac{1}{2} g^{rs}_{pq} \{E^{pq}_{rs}\}

    Examples:
        If required is active:+1 and virtual:-1, this keeps only
        two-body Hamiltonian insertions whose net action supplies that
        balance.
    """
    out = []

    for p, q, r, s in allGeneralH2Indices():
        if not sameBalance(
            h2Balance(p, q, r, s),
            required,
        ):
            continue

        out.append((
            scale(
                h2Coeff(p, q, r, s),
                Fraction(1, 2),
            ),
            groupE2(p, q, r, s, groupId),
        ))

    return tuple(out)

def hTermsWithBalance(
    groupId: int,
    required: dict[Space, int],
) -> tuple[tuple[Expr, Group], ...]:
    """
    Build Hamiltonian terms with a required balance.

    Notation:
        H = H_1 + H_2

    Examples:
        hTermsWithBalance(1, required) returns only Hamiltonian terms
        that can make the residual Wick product fully contractible.
    """
    return (
        h1TermsWithBalance(
            groupId,
            required,
        )
        + h2TermsWithBalance(
            groupId,
            required,
        )
    )

def tCoeff(spec: ExcitationSpec) -> Expr:
    """
    Build the cluster-amplitude coefficient for one excitation class.

    Notation:
        t^q_p \\{E^p_q\\}
        t^{rs}_{pq} \\{E^{pq}_{rs}\\}

    Examples:
        For C -> A, returns t^i_u.
        For CA -> AV, returns t^{iu}_{va}.
    """
    if len(spec.creators) == 1:
        return tensor(
            "t1",
            spec.annihilators,
            spec.creators,
        )

    if len(spec.creators) == 2:
        return tensor(
            "t2",
            spec.annihilators,
            spec.creators,
        )

    raise ValueError(f"unsupported excitation rank {len(spec.creators)}")

def tTerms(groupId: int) -> tuple[tuple[Expr, Group], ...]:
    """
    Build all cluster-operator terms.

    Notation:
        T =
        \sum_\nu t_\nu \tau_\nu = T_1 + T_2

    Examples:
        One-body excitation classes contribute t1 amplitudes.
        Two-body excitation classes contribute amplitudes with the
        cluster prefactor 1/2.
    """
    out = []

    for spec in EXCITATIONS.values():
        coeff = tCoeff(spec)

        if len(spec.creators) == 2:
            coeff = scale(
                coeff,
                Fraction(1, 2),
            )

        out.append((
            coeff,
            ketGroup(spec, groupId),
        ))

    return tuple(out)

def tTermsWithBalance(
    groupId: int,
) -> tuple[tuple[Expr, Group, dict[Space, int]], ...]:
    """
    Build all cluster-operator terms with orbital-space balances.

    Notation:
        T = \sum_\nu t_\nu \tau_\nu

    Examples:
        C -> A contributes t^i_u \{E^u_i\} with active:+1 and core:-1.
        Double excitations carry the cluster prefactor 1/2.
    """
    out = []

    for spec in EXCITATIONS.values():
        coeff = tCoeff(spec)

        if len(spec.creators) == 2:
            coeff = scale(
                coeff,
                Fraction(1, 2),
            )

        out.append((
            coeff,
            ketGroup(spec, groupId),
            excitationBalance(spec),
        ))

    return tuple(out)

def contractedContribution(
    wick: Wick,
    coeffs: tuple[Expr, ...],
    groups: tuple[Group, ...],
) -> Expr:
    """
    Evaluate one coefficient-weighted Wick product.

    Notation:
        c_1 c_2 ... c_n = \langle \Phi | \{G_1\}\{G_2\} \cdots \{G_n\} | \Phi \rangle

    Examples:
        contractedContribution(wick, (f, t), (bra, h, tau))
        evaluates one first-order residual contribution.
    """
    value = wick.eval(
        Product(groups)
    )

    if not value:
        return ()

    out = value

    for coeff in reversed(coeffs):
        out = mul(
            coeff,
            out,
        )

    return out

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
        = \langle \Phi | \tau_\mu^\dagger H | \Phi \rangle_c

        H = \sum_{p \in A \cup V} \sum_{q \in C \cup A} f^q_p \{E^p_q\}
        + \frac{1}{2} \sum_{p,q \in A \cup V} \sum_{r,s \in C \cup A} g^{rs}_{pq} \{E^{pq}_{rs}\}

    Examples:
        r0Expr("CtoA") evaluates the zeroth-order residual for the C -> A
        excitation class, if "CtoA" is present in EXCITATIONS.
    """
    spec = EXCITATIONS[name]
    wick = Wick(Ref(maxActiveCumulantRank = 4))

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

def r1Expr(name: str) -> Expr:
    """
    Evaluate one first-order residual contribution.

    Notation:
        R_mu^{(1)}
        =
        \langle \Phi | \tau_mu^\dagger H T | \Phi \rangle_c

    Examples:
        r1Expr("CToA") evaluates the terms linear in the cluster amplitudes
        for the C -> A residual.
    """
    spec = EXCITATIONS[name]
    wick = Wick(Ref(maxActiveCumulantRank = 4))
    bra = braGroup(spec, 0)
    braBalance = excitationBalance(
        spec,
        daggered = True,
    )
    terms = []

    for tCoeffExpr, tGroup, tBalance in tTermsWithBalance(2):
        required = negateBalance(
            mergeBalances(
                braBalance,
                tBalance,
            )
        )

        for hCoeffExpr, hGroup in hTermsWithBalance(1, required):
            contribution = contractedContribution(
                wick,
                (
                    hCoeffExpr,
                    tCoeffExpr,
                ),
                (
                    bra,
                    hGroup,
                    tGroup,
                ),
            )

            terms.extend(contribution)

    return canonicaliseForOutput(
        combine(tuple(terms))
    )

def residualExpr(name: str, order: int = 0) -> Expr:
    """
    Evaluate one residual contribution by cluster-amplitude order.

    Notation:
        R_mu = R_mu^{(0)} + R_mu^{(1)} + R_mu^{(2)}

    Examples:
        residualExpr("CToA", 0) evaluates the direct residual.
        residualExpr("CToA", 1) evaluates terms linear in T.
    """
    if order == 0:
        return r0Expr(name)

    if order == 1:
        return r1Expr(name)

    raise ValueError(f"unsupported residual order {order}")
