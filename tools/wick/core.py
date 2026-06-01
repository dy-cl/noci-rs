# tools/wick/core.py
# Minimal object rule Wick/GNO evaluation tool.
# Objects:
#   Op: Fermionic operators a^\dagger_{p \sigma}, a_{p \sigma}.
#   Group: Nomral ordered object inside braces \{ E_i^u \}.
#   Product: Product of normal ordered Group \{ E_i^u \} \{ E_a^i \}.
#   Ref: Gives the reference expectation of Fermionic operators \langle \Phi | a_i^\dagger a_j | \Phi \rangle.
#   Spin: Performs spin summation.
#   Wick: Generalised Wick evaluator.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import cache
from itertools import permutations, product as cartesianProduct

class Space(Enum):
    """Orbital reference space.

    Notation:
        i,j,k,l,... in C      core, occupied in the reference
        u,v,w,x,... in A      active
        a,b,c,d,... in V      virtual, empty in the reference

    Examples:
        Idx("i", Space.CORE)    represents i in C
        Idx("u", Space.ACTIVE)  represents u in A
        Idx("a", Space.VIRTUAL) represents a in V
    """
    CORE = "C"
    ACTIVE = "A"
    VIRTUAL = "V"

ALPHA = "alpha"
BETA = "beta"
SPINS = (ALPHA, BETA)

@dataclass(frozen = True, order = True)
class Idx:
    """Spin-free orbital index.

    Notation:
        p, q, r, s are general orbital indices.
        i, j are core indices.
        u, v are active indices.
        a, b are virtual indices.

    Examples:
        Idx("u", Space.ACTIVE) is active index u.
    """
    name: str
    space: Space

@dataclass(frozen = True, order = True)
class Op:
    """Spin-orbital fermion operator.

    Notation:
        a^\dagger_{p\sigma} creation operator
        a_{p\sigma} annihilation operator

    Fields:
        kind  = "create" or "annihilate"
        idx   = spin-free orbital index p
        spin  = "alpha" or "beta"
        group = integer id of the GNO group this operator belongs to

    Examples:
        Op("create", Idx("u", Space.ACTIVE), "alpha", 0) is a^\dagger_{u\alpha}
        Op("annihilate", Idx("v", Space.ACTIVE), "alpha", 0) is a_{v\alpha}

    If both have group = 0, then they are inside the same normal-ordered group:

        \{ a^\dagger_{u\alpha} a_{v\alpha} \}
    """
    kind: str
    idx: Idx
    spin: str
    group: int

@dataclass(frozen = True)
class Group:
    """One generalized normal ordered group.

    Notation:
        \{E^p_q\}
        \{E^{pq}_{rs}\}

    Examples:
        \{E^u_v\}
        =
        \{a^\dagger_{u\alpha}a_{v\alpha}\}
        +
        \{a^\dagger_{u\beta}a_{v\beta}\}
    """
    strings: tuple[tuple[Op, ...], ...]

@dataclass(frozen = True)
class Product:
    """Product of GNO groups.

    Notation:
        \{E^p_q\}\{E^{rs}_{tu}\}

    Example:
        Product((tau1(u, v), tau1(x, w))) is \{E^u_v\}\{E^x_w\}
    """
    groups: tuple[Group, ...]

@dataclass(frozen = True, order = True)
class Delta:
    """Kronecker delta.

    Notation:
        \delta^p_q

    Example:
        Delta(i, j) is \delta^i_j
    """
    left: Idx
    right: Idx

@dataclass(frozen = True, order = True)
class Tensor:
    """Spin-free tensor factor.

    Notation:
        \Gamma^u_v
        \Theta^u_v
        \Lambda^{ux}_{vw}
        \Lambda^{tyz}_{uwx}
        \Lambda^{prtv}_{qsuw}

    Examples:
        Tensor("Gamma1", (u,), (v,)) represents \Gamma^u_v.
        Tensor("Theta", (u,), (v,)) represents \Theta^u_v.
        Tensor("Lambda2", (u, x), (v, w)) represents \Lambda^{ux}_{vw}.
        Tensor("Lambda3", (t, y, z), (u, w, x)) represents \Lambda^{tyz}_{uwx}.
        Tensor("Lambda4", (p, r, t, v), (q, s, u, w)) represents \Lambda^{prtv}_{qsuw}.
    """
    name: str
    upper: tuple[Idx, ...]
    lower: tuple[Idx, ...]

@dataclass(frozen = True, order = True)
class Term:
    """One scalar product of deltas and tensors.

    Notation:
        c \delta ... \Gamma ... \Lambda ...

    Examples:
        Term(
            coeff = Fraction(1, 2),
            deltas = (Delta(i, j),),
            tensors = (Tensor("Gamma1", (u,), (w,)), Tensor("Theta", (x,), (v,))),
        )

        represents

            (1/2) \delta^i_j \Gamma^u_w \Theta^x_v
    """
    coeff: Fraction = Fraction(1)
    deltas: tuple[Delta, ...] = ()
    tensors: tuple[Tensor, ...] = ()

Expr = tuple[Term, ...]

def zero() -> Expr:
    """Return the zero expression.

    Notation:
        0

    Examples:
        A forbidden or vanishing contraction returns zero.

            \kappa(a^\dagger_{a\alpha}, a_{b\alpha}) = 0

        for virtual orbitals.
    """
    return ()

def one() -> Expr:
    """Return scalar one.

    Notation:
        1

    Examples:
        The empty product is one.

            product over no factors = 1
    """
    return (Term(),)

def delta(p: Idx, q: Idx) -> Expr:
    """Return a Kronecker delta expression.

    Notation:
        \delta^p_q

    Examples:
        delta(i, j) represents \delta^i_j.
        delta(a, b) represents \delta^a_b.
    """
    return (Term(deltas = (Delta(p, q),)),)

def tensor(name: str, upper: tuple[Idx, ...], lower: tuple[Idx, ...]) -> Expr:
    """Return one spin-free tensor expression.

    Notation:
        \Gamma, \Theta, \Lambda_2, \Lambda_3, \Lambda_4

    Examples:
        tensor("Gamma1", (u,), (v,)) represents \Gamma^u_v.
        tensor("Theta", (u,), (v,)) represents \Theta^u_v.
        tensor("Lambda2", (u, x), (v, w)) represents \Lambda^{ux}_{vw}.
        tensor("Lambda4", (p, r, t, v), (q, s, u, w)) represents \Lambda^{prtv}_{qsuw}.
    """
    return (Term(tensors = (Tensor(name, upper, lower),)),)

def scale(expr: Expr, coeff: int | Fraction) -> Expr:
    """Scale an expression.

    Notation:
        c A

    Examples:
        scale(tensor("Lambda2", (u, x), (v, w)), Fraction(-1, 2))

        represents

            -1/2 \Lambda^{ux}_{vw}
    """
    c = coeff if isinstance(coeff, Fraction) else Fraction(coeff)

    if c == 0:
        return zero()

    return tuple(
        Term(
            coeff = c * term.coeff,
            deltas = term.deltas,
            tensors = term.tensors,
        )
        for term in expr
        if c * term.coeff != 0
    )

def add(*exprs: Expr) -> Expr:
    """Add expressions and combine like terms.

    Notation:
        A + B + ...

    Examples:
        add(
            tensor("Gamma1", (u,), (v,)),
            tensor("Lambda2", (u, x), (v, w)),
        )

        represents

            \Gamma^u_v + \Lambda^{ux}_{vw}
    """
    terms = []

    for expr in exprs:
        terms.extend(expr)

    return combine(tuple(terms))

def mul(left: Expr, right: Expr) -> Expr:
    """Multiply scalar expressions commutatively.

    Notation:
        A B

    Examples:
        mul(delta(i, j), tensor("Gamma1", (u,), (w,)))

        represents

            \delta^i_j \Gamma^u_w
    """
    out = []

    for a in left:
        for b in right:
            out.append(
                Term(
                    coeff = a.coeff * b.coeff,
                    deltas = tuple(sorted(a.deltas + b.deltas)),
                    tensors = tuple(sorted(a.tensors + b.tensors)),
                )
            )

    return combine(tuple(out))

def prod(exprs: tuple[Expr, ...]) -> Expr:
    """Multiply many scalar expressions.

    Notation:
        product_k A_k

    Examples:
        prod((
            delta(i, j),
            tensor("Gamma1", (u,), (w,)),
            tensor("Theta", (x,), (v,)),
        ))

        represents

            \delta^i_j \Gamma^u_w \Theta^x_v
    """
    out = one()

    for expr in exprs:
        out = mul(out, expr)

    return out

def combine(expr: Expr) -> Expr:
    """Combine identical scalar products.

    Notation:
        c A + d A = (c + d) A

    Examples:
        (1/2) \Gamma^u_v + (1/2) \Gamma^u_v

        becomes

            \Gamma^u_v
    """
    acc: dict[tuple, Fraction] = {}

    for term in expr:
        key = (term.deltas, term.tensors)
        acc[key] = acc.get(key, Fraction(0)) + term.coeff

    return tuple(
        Term(
            coeff = coeff,
            deltas = deltas,
            tensors = tensors,
        )
        for (deltas, tensors), coeff in sorted(acc.items())
        if coeff != 0
    )

def anticommutator(left: Op, right: Op) -> Expr:
    """Canonical anticommutation relation.

    Notation:
        \{a_{p\sigma}, a^\dagger_{q\tau}\}
        =
        \delta^p_q \delta_{\sigma\tau}

    Examples:
        anticommutator(a_{p\alpha}, a^\dagger_{q\alpha})
        gives \delta^p_q.

        anticommutator(a_{p\alpha}, a^\dagger_{q\beta})
        gives 0.

        anticommutator(a^\dagger_{p\alpha}, a^\dagger_{q\alpha})
        gives 0.

        anticommutator(a_{p\alpha}, a_{q\alpha})
        gives 0.
    """
    if left.kind != "annihilate" or right.kind != "create":
        return zero()

    if left.spin != right.spin:
        return zero()

    if left.idx.space != right.idx.space:
        return zero()

    return delta(left.idx, right.idx)

def permutationSign(sequence: tuple[int, ...]) -> int:
    """Return fermionic parity of a permutation.

    Notation:
        Every swap of fermion operators contributes a factor of -1.

    Examples:
        permutationSign((1, 0)) gives -1 because

            a_p a_q = -a_q a_p

        permutationSign((0, 2, 1, 3)) gives -1 because the sequence has one
        inversion.
    """
    inversions = 0

    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            if sequence[i] > sequence[j]:
                inversions += 1

    return -1 if inversions % 2 else 1

@cache
def partitions(items: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return set partitions of operator positions.

    Notation:
        Generalised Wick theorem sums over partitions P:

            < \{A\}\{B\}... >
            =
            sum_P sign(P) product_{block in P} \kappa(block)

    Examples:
        items = (0, 1, 2, 3)

        One partition is

            ((0, 2), (1, 3))

        representing

            \kappa(op_0, op_2) \kappa(op_1, op_3)
    """
    if not items:
        return ((),)

    first = items[0]
    rest = items[1:]
    out = []

    for partition in partitions(rest):
        out.append(((first,),) + partition)

        for i, block in enumerate(partition):
            out.append(
                partition[:i]
                + (tuple(sorted((first,) + block)),)
                + partition[i + 1:]
            )

    canonical = {
        tuple(sorted(partition, key = lambda block: block[0]))
        for partition in out
    }

    return tuple(sorted(canonical))

def partitionSign(partition: tuple[tuple[int, ...], ...]) -> int:
    """Return fermion sign for collecting Wick blocks.

    Notation:
        sign(P) is the parity of the flattened block order.

    Examples:
        partition = ((0, 2), (1, 3))

        Flattened order:

            (0, 2, 1, 3)

        There is one inversion, so sign(P) = -1.
    """
    sequence = tuple(
        i
        for block in partition
        for i in block
    )

    return permutationSign(sequence)

class Spin:
    """Spin-orbital to spin-free projection rules.

    Notation:
        gamma^{p_\sigma}_{q_\tau} -> \Gamma^p_q

        lambda^{p_\sigma q_\tau}_{r_\mu s_\nu}
        -> \Lambda^{pq}_{rs}

        lambda_3 -> \Lambda_3

        lambda_4 -> \Lambda_4

    Examples:
        gamma^{u_\alpha}_{v_\alpha}
        -> (1/2) \Gamma^u_v

        lambda^{p_\alpha q_\beta}_{r_\alpha s_\beta}
        -> (1/6)(2 \Lambda^{pq}_{rs} + \Lambda^{pq}_{sr})
    """

    @staticmethod
    def projectCumulant(
        rank: int,
        upper: tuple[Idx, ...],
        lower: tuple[Idx, ...],
        upperSpins: tuple[str, ...],
        lowerSpins: tuple[str, ...],
    ) -> Expr:
        """Project one spin-orbital cumulant component.

        Notation:
            spin-orbital cumulant -> spin-free cumulant combination

        Examples:
            projectCumulant(1, (u,), (v,), ("alpha",), ("alpha",))

            represents

                gamma^{u_\alpha}_{v_\alpha}
                -> (1/2) \Gamma^u_v

            projectCumulant(2, (u, x), (v, w), ("alpha", "beta"), ("alpha", "beta"))

            represents

                lambda^{u_\alpha x_\beta}_{v_\alpha w_\beta}
                -> (1/6)(2 \Lambda^{ux}_{vw} + \Lambda^{ux}_{wv})

            projectCumulant(4, (p, r, t, v), (q, s, u, w), spins, spins)

            represents

                lambda^{p_{\sigma_1} r_{\sigma_2} t_{\sigma_3} v_{\sigma_4}}
                      _{q_{\sigma_1} s_{\sigma_2} u_{\sigma_3} w_{\sigma_4}}
                -> sum_{\pi in S_4} c_\pi \Lambda^{prtv}_{\pi(qsuw)}
        """
        if rank == 1:
            return Spin.gamma1(
                upper,
                lower,
                upperSpins,
                lowerSpins,
            )

        names = {
            2: "Lambda2",
            3: "Lambda3",
            4: "Lambda4",
        }

        if rank not in names:
            raise NotImplementedError(f"rank {rank} active cumulant unsupported")

        return Spin.lambdaK(
            names[rank],
            upper,
            lower,
            upperSpins,
            lowerSpins,
        )

    @staticmethod
    def gamma1(
        upper: tuple[Idx, ...],
        lower: tuple[Idx, ...],
        upperSpins: tuple[str, ...],
        lowerSpins: tuple[str, ...],
    ) -> Expr:
        """Rank-1 spin replacement.

        Notation:
            gamma^{p_\sigma}_{q_\tau}

        Examples:
            gamma^{u_\alpha}_{v_\alpha}
            =
            (1/2) \Gamma^u_v

            gamma^{u_\alpha}_{v_\beta}
            =
            0
        """
        if upperSpins[0] != lowerSpins[0]:
            return zero()

        return scale(
            tensor("Gamma1", upper, lower),
            Fraction(1, 2),
        )

    @staticmethod
    def lambdaK(
        name: str,
        upper: tuple[Idx, ...],
        lower: tuple[Idx, ...],
        upperSpins: tuple[str, ...],
        lowerSpins: tuple[str, ...],
    ) -> Expr:
        """Rank-k spin-free cumulant projection.

        Notation:
            lambda^{p_1 sigma_1 ... p_k sigma_k}_{q_1 tau_1 ... q_k tau_k}
            =
            sum_{\pi in S_k} c_\pi \Lambda^{p_1...p_k}_{\pi(q_1...q_k)}

        Examples:
            lambda^{p_\alpha q_\alpha}_{r_\alpha s_\alpha}
            =
            (1/6)(\Lambda^{pq}_{rs} - \Lambda^{pq}_{sr})

            lambda^{p_\alpha q_\beta}_{r_\alpha s_\beta}
            =
            (1/6)(2 \Lambda^{pq}_{rs} + \Lambda^{pq}_{sr})

            lambda^{t_\alpha y_\beta z_\alpha}_{u_\alpha w_\beta x_\alpha}
            =
            sum_{\pi in S_3} c_\pi \Lambda^{tyz}_{\pi(uwx)}

            lambda^{p_\alpha r_\beta t_\alpha v_\beta}_{q_\alpha s_\beta u_\alpha w_\beta}
            =
            sum_{\pi in S_4} c_\pi \Lambda^{prtv}_{\pi(qsuw)}
        """
        out = zero()

        for lowerPerm, coeff in Spin.coeffs(
            len(upper),
            upperSpins,
            lowerSpins,
        ):
            out = add(
                out,
                scale(
                    tensor(
                        name,
                        upper,
                        tuple(lower[i] for i in lowerPerm),
                    ),
                    coeff,
                ),
            )

        return out

    @staticmethod
    def coeffs(
        rank: int,
        upperSpins: tuple[str, ...],
        lowerSpins: tuple[str, ...],
    ) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
        """Return lower-permutation spin-replacement coefficients.

        Notation:
            lambda^{p_1 sigma_1 ... p_k sigma_k}_{q_1 tau_1 ... q_k tau_k}
            =
            sum_{\pi in S_k} c_\pi Lambda^{p_1...p_k}_{q_{\pi(1)}...q_{\pi(k)}}

        Examples:
            rank = 2 reproduces

                lambda^{p_\alpha q_\beta}_{r_\alpha s_\beta}
                =
                (1/6)(2 Lambda^{pq}_{rs} + Lambda^{pq}_{sr})

            rank = 3 gives one deterministic gauge representative for

                lambda^{p_\sigma q_\tau r_\mu}_{s_\nu t_\rho u_\eta}
                =
                sum_{\pi in S_3} c_\pi Lambda^{pqr}_{\pi(stu)}

        The rank-3 system is singular because the spin space has only two
        spin functions. Free variables are set to zero, which fixes a gauge.
        """
        if rank not in (2, 3, 4):
            raise NotImplementedError(f"rank {rank} spin replacement coefficients are not implemented")

        if len(upperSpins) != rank or len(lowerSpins) != rank:
            raise ValueError("spin tuples do not match cumulant rank")

        perms = tuple(permutations(range(rank)))

        def invPerm(p: tuple[int, ...]) -> tuple[int, ...]:
            out = [0] * len(p)

            for i, x in enumerate(p):
                out[x] = i

            return tuple(out)

        def compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(p[q[i]] for i in range(len(p)))

        def cycleCount(p: tuple[int, ...]) -> int:
            seen = [False] * len(p)
            cycles = 0

            for i in range(len(p)):
                if seen[i]:
                    continue

                cycles += 1
                j = i

                while not seen[j]:
                    seen[j] = True
                    j = p[j]

            return cycles

        def solveConsistent(mat: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
            nRows = len(mat)
            nCols = len(mat[0])
            aug = [row[:] + [rhs[i]] for i, row in enumerate(mat)]

            pivotCols = []
            row = 0

            for col in range(nCols):
                pivot = None

                for r in range(row, nRows):
                    if aug[r][col] != 0:
                        pivot = r
                        break

                if pivot is None:
                    continue

                if pivot != row:
                    aug[row], aug[pivot] = aug[pivot], aug[row]

                scaleFactor = aug[row][col]
                aug[row] = [x / scaleFactor for x in aug[row]]

                for r in range(nRows):
                    if r == row:
                        continue

                    factor = aug[r][col]

                    if factor == 0:
                        continue

                    aug[r] = [
                        aug[r][i] - factor * aug[row][i]
                        for i in range(nCols + 1)
                    ]

                pivotCols.append(col)
                row += 1

                if row == nRows:
                    break

            for r in range(row, nRows):
                if all(aug[r][c] == 0 for c in range(nCols)) and aug[r][-1] != 0:
                    raise ValueError("inconsistent spin-projection system")

            sol = [Fraction(0) for _ in range(nCols)]

            for r, col in enumerate(pivotCols):
                sol[col] = aug[r][-1]

            return sol

        gram = []

        for p in perms:
            row = []

            for q in perms:
                rel = compose(invPerm(p), q)
                row.append(
                    Fraction(
                        permutationSign(p)
                        * permutationSign(q)
                        * (2 ** cycleCount(rel)),
                        1,
                    )
                )

            gram.append(row)

        rhs = []

        for p in perms:
            allowed = all(
                upperSpins[i] == lowerSpins[p[i]]
                for i in range(rank)
            )

            rhs.append(
                Fraction(permutationSign(p), 1)
                if allowed
                else Fraction(0)
            )

        sol = solveConsistent(gram, rhs)

        return tuple(
            (p, c)
            for p, c in zip(perms, sol)
            if c != 0
        )

class Ref:
    """Reference-state connected cumulant rules.

    Notation:
        \kappa(op_1, op_2, ...)

    Examples:
        \kappa(a^\dagger_{i\alpha}, a_{j\alpha}) = \delta^i_j

        \kappa(a_{a\alpha}, a^\dagger_{b\alpha}) = \delta^a_b

        \kappa(active string) gives \Gamma, \Lambda_2, \Lambda_3, or \Lambda_4.
    """

    def kappa(self, ops: tuple[Op, ...]) -> Expr:
        """Connected cumulant of one Wick block.

        Notation:
            \kappa(B)

        Examples:
            kappa(a^\dagger_{i\alpha}, a_{j\alpha})
            =
            \delta^i_j

            kappa(a_{a\alpha}, a^\dagger_{b\alpha})
            =
            \delta^a_b

            kappa(a^\dagger_{u\alpha}, a_{v\alpha})
            =
            gamma^{u_\alpha}_{v_\alpha}
            ->
            (1/2) \Gamma^u_v

            kappa(a^\dagger_{u\alpha}, a^\dagger_{x\beta}, a_{w\beta}, a_{v\alpha})
            =
            lambda^{u_\alpha x_\beta}_{v_\alpha w_\beta}
            ->
            spin-free \Lambda_2 combination
        """
        if not ops:
            return zero()

        spaces = {op.idx.space for op in ops}

        if spaces == {Space.ACTIVE}:
            return self.activeKappa(ops)

        if len(ops) == 2:
            return self.frozenPair(ops[0], ops[1])

        return zero()

    def frozenPair(self, left: Op, right: Op) -> Expr:
        """Core/virtual deterministic cumulants.

        Notation:
            \kappa(a^\dagger_{i\sigma}, a_{j\tau})
            =
            \delta^i_j \delta_{\sigma\tau}

            \kappa(a_{a\sigma}, a^\dagger_{b\tau})
            =
            \delta^a_b \delta_{\sigma\tau}

        Examples:
            frozenPair(a^\dagger_{i\alpha}, a_{j\alpha})
            =
            \delta^i_j

            frozenPair(a_{a\alpha}, a^\dagger_{b\alpha})
            =
            \delta^a_b

            frozenPair(a_{i\alpha}, a^\dagger_{j\alpha})
            =
            0

            frozenPair(a^\dagger_{a\alpha}, a_{b\alpha})
            =
            0
        """
        if left.spin != right.spin:
            return zero()

        if left.idx.space != right.idx.space:
            return zero()

        if left.idx.space == Space.CORE:
            if left.kind == "create" and right.kind == "annihilate":
                return delta(left.idx, right.idx)

            return zero()

        if left.idx.space == Space.VIRTUAL:
            if left.kind == "annihilate" and right.kind == "create":
                return delta(left.idx, right.idx)

            return zero()

        return zero()

    def activeKappa(self, ops: tuple[Op, ...]) -> Expr:
        """Active connected cumulant.

        Notation:
            active \kappa -> spin-orbital RDM/cumulant -> spin-free tensor

        Examples:
            \kappa(a^\dagger_{u\alpha}, a_{v\alpha})
            =
            gamma^{u_\alpha}_{v_\alpha}
            ->
            (1/2) \Gamma^u_v

            \kappa(a_{u\alpha}, a^\dagger_{v\alpha})
            =
            eta^{v_\alpha}_{u\alpha}
            ->
            (1/2) \Theta^v_u
        """
        if len(ops) == 2:
            left, right = ops

            if left.spin != right.spin:
                return zero()

            if left.kind == "create" and right.kind == "annihilate":
                return scale(
                    tensor("Gamma1", (left.idx,), (right.idx,)),
                    Fraction(1, 2),
                )

            if left.kind == "annihilate" and right.kind == "create":
                return scale(
                    tensor("Theta", (right.idx,), (left.idx,)),
                    Fraction(1, 2),
                )

            return zero()

        normal = self.normalOrder(ops)

        if normal is None:
            return zero()

        sign, creators, annihilators = normal
        rank = len(creators)

        if rank != len(annihilators):
            return zero()

        upper = tuple(op.idx for op in creators)
        upperSpins = tuple(op.spin for op in creators)

        lower = tuple(op.idx for op in reversed(annihilators))
        lowerSpins = tuple(op.spin for op in reversed(annihilators))

        return scale(
            Spin.projectCumulant(
                rank,
                upper,
                lower,
                upperSpins,
                lowerSpins,
            ),
            sign,
        )

    def normalOrder(self, ops: tuple[Op, ...]):
        """Bring active operators to creator-left form.

        Notation:
            a_{u\alpha} a^\dagger_{v\alpha}
            =
            \delta^u_v - a^\dagger_{v\alpha} a_{u\alpha}
        """
        creators = tuple(op for op in ops if op.kind == "create")
        annihilators = tuple(op for op in ops if op.kind == "annihilate")

        if len(creators) + len(annihilators) != len(ops):
            return None

        inversions = 0

        for i, left in enumerate(ops):
            if left.kind != "annihilate":
                continue

            for right in ops[i + 1:]:
                if right.kind == "create":
                    inversions += 1

        sign = -1 if inversions % 2 else 1

        return sign, creators, annihilators

class Wick:
    """Generalised Wick evaluator.

    Notation:
        < \{A\}\{B\}\{C\}... >
        =
        sum_{valid partitions P}
            sign(P) product_{block in P} \kappa(block)

    Examples:
        Wick(ref).eval(Product((tau1(i, u, 0), tau1(x, a, 1))))

        represents

            <Phi| \{E^i_u\}\{E^x_a\} |Phi>
    """

    def __init__(self, ref: Ref):
        self.ref = ref

    def eval(self, product: Product) -> Expr:
        """Evaluate a product of normal ordered groups.

        Notation:
            <Phi| product |Phi>

        Examples:
            product = Product((tau2(i, u, v, a, 0), tau2(x, b, j, w, 1)))

            represents

                <Phi| \{E^{iu}_{va}\}\{E^{xb}_{jw}\} |Phi>

            The algorithm expands spin-free E operators into spin-orbital
            strings, enumerates Wick partitions, rejects internal contractions,
            applies Ref.kappa to each block, and sums the result.
        """
        out = zero()

        for ops in self.expand(product):
            out = add(
                out,
                self.evalSpinString(ops),
            )

        return simplify(out)

    def expand(self, product: Product) -> tuple[tuple[Op, ...], ...]:
        """Expand spin-free groups into spin-orbital strings.

        Notation:
            \{E^u_v\}
            =
            \{a^\dagger_{u\alpha}a_{v\alpha}\}
            +
            \{a^\dagger_{u\beta}a_{v\beta}\}

            \{E^{ua}_{iv}\}
            =
            sum_{\sigma\tau}
            \{a^\dagger_{u\sigma}a^\dagger_{a\tau}a_{v\tau}a_{i\sigma}\}

        Examples:
            Product((tau1(u, v, 0), tau1(x, w, 1)))

            expands into four spin strings.
        """
        strings = ((),)

        for group in product.groups:
            nextStrings = []

            for prefix in strings:
                for suffix in group.strings:
                    nextStrings.append(prefix + suffix)

            strings = tuple(nextStrings)

        return strings

    def evalSpinString(self, ops: tuple[Op, ...]) -> Expr:
        """Evaluate one spin-orbital string by generalized Wick's theorem.

        Notation:
            sum_P sign(P) product_{block in P} \kappa(block)

        Examples:
            Input string:

                \{a^\dagger_{i\alpha}a_{u\alpha}\}
                \{a^\dagger_{x\alpha}a_{a\alpha}\}

            The block

                \kappa(a^\dagger_{i\alpha}, a_{u\alpha})

            is rejected because it is internal to one group.

            The block

                \kappa(a_{u\alpha}, a^\dagger_{x\alpha})

            is allowed if it connects different groups.
        """
        out = zero()
        positions = tuple(range(len(ops)))

        for partition in partitions(positions):
            factors = []
            valid = True

            for block in partition:
                if self.internal(ops, block):
                    valid = False
                    break

                value = self.ref.kappa(
                    tuple(ops[i] for i in block)
                )

                if not value:
                    valid = False
                    break

                factors.append(value)

            if not valid:
                continue

            out = add(
                out,
                scale(
                    prod(tuple(factors)),
                    self.sign(partition),
                ),
            )

        return out

    def internal(self, ops: tuple[Op, ...], block: tuple[int, ...]) -> bool:
        """Check whether a Wick block is internal to one group.

        Notation:
            Internal contractions inside \{...\} are forbidden.

        Examples:
            In

                \{a^\dagger_{u\alpha}a_{v\alpha}\}

            the block

                (a^\dagger_{u\alpha}, a_{v\alpha})

            is internal and is rejected.

            In

                \{a^\dagger_{u\alpha}a_{v\alpha}\}
                \{a^\dagger_{x\alpha}a_{w\alpha}\}

            the block

                (a_{v\alpha}, a^\dagger_{x\alpha})

            connects two groups and is allowed.
        """
        groups = {
            ops[i].group
            for i in block
        }

        return len(groups) == 1

    def sign(self, partition: tuple[tuple[int, ...], ...]) -> int:
        """Fermion sign for a Wick partition.

        Notation:
            sign(P) is the parity of the flattened block ordering.

        Examples:
            partition = ((0, 2), (1, 3))

            flattened sequence:

                (0, 2, 1, 3)

            has one inversion, so sign(P) = -1.
        """
        return partitionSign(partition)

def groupE1(p: Idx, q: Idx, groupId: int) -> Group:
    """Build spin components of spin-free E^p_q.

    Notation:
        E^p_q = sum_\sigma a^\dagger_{p\sigma} a_{q\sigma}

    Examples:
        groupE1(u, v, 0)

        represents

            E^u_v
            =
            a^\dagger_{u\alpha}a_{v\alpha}
            +
            a^\dagger_{u\beta}a_{v\beta}

        If used as a GNO group, this is \{E^u_v\}.
    """
    return Group(
        tuple(
            (
                Op("create", p, spin, groupId),
                Op("annihilate", q, spin, groupId),
            )
            for spin in SPINS
        )
    )

def groupE2(p: Idx, q: Idx, r: Idx, s: Idx, groupId: int) -> Group:
    """Build spin components of spin-free E^{pq}_{rs}.

    Notation:
        E^{pq}_{rs}
        =
        sum_{\sigma\tau}
        a^\dagger_{p\sigma} a^\dagger_{q\tau} a_{s\tau} a_{r\sigma}

    Examples:
        groupE2(u, a, i, v, 0)

        represents

            E^{ua}_{iv}
            =
            sum_{\sigma\tau}
            a^\dagger_{u\sigma} a^\dagger_{a\tau} a_{v\tau} a_{i\sigma}

        If used as a GNO group, this is \{E^{ua}_{iv}\}.
    """
    return Group(
        tuple(
            (
                Op("create", p, sigma, groupId),
                Op("create", q, tau, groupId),
                Op("annihilate", s, tau, groupId),
                Op("annihilate", r, sigma, groupId),
            )
            for sigma in SPINS
            for tau in SPINS
        )
    )

def tau1(create: Idx, annihilate: Idx, groupId: int) -> Group:
    """One-body GNO excitation.

    Notation:
        \{E^p_q\}

    Examples:
        tau1(u, i, 0)

        represents

            \{E^u_i\}

        This is a C -> A excitation.
    """
    return groupE1(create, annihilate, groupId)

def tau2(
    create1: Idx,
    create2: Idx,
    annihilate1: Idx,
    annihilate2: Idx,
    groupId: int,
) -> Group:
    """Two-body GNO excitation.

    Notation:
        \{E^{pq}_{rs}\}

    Examples:
        tau2(u, a, i, v, 0)

        represents

            \{E^{ua}_{iv}\}

        This is a CA -> AV excitation.
    """
    return groupE2(
        create1,
        create2,
        annihilate1,
        annihilate2,
        groupId,
    )

def daggerTau1(create: Idx, annihilate: Idx, groupId: int) -> Group:
    """Adjoint of a one-body excitation.

    Notation:
        (\{E^p_q\})^\dagger = \{E^q_p\}

    Examples:
        daggerTau1(u, i, 0)

        represents

            (\{E^u_i\})^\dagger = \{E^i_u\}
    """
    return tau1(
        annihilate,
        create,
        groupId,
    )

def daggerTau2(
    create1: Idx,
    create2: Idx,
    annihilate1: Idx,
    annihilate2: Idx,
    groupId: int,
) -> Group:
    """Adjoint of a two-body excitation.

    Notation:
        (\{E^{pq}_{rs}\})^\dagger = \{E^{rs}_{pq}\}

    Examples:
        daggerTau2(u, a, i, v, 0)

        represents

            (\{E^{ua}_{iv}\})^\dagger = \{E^{iv}_{ua}\}
    """
    return tau2(
        annihilate1,
        annihilate2,
        create1,
        create2,
        groupId,
    )

def simplify(expr: Expr) -> Expr:
    """Algebraic simplification only.

    Notation:
        Apply identities that are true independently of any overlap block.

    Examples:
        \delta^i_j \delta^j_k -> \delta^i_k

        \Gamma^i_j -> 2 \delta^i_j

        \Gamma^a_b -> 0

        \Theta^u_v = \delta^u_v - \Gamma^u_v

        \Lambda^{pq}_{rs} = 0 if any index is not active

        \Lambda^{ux}_{vw} = \Lambda^{xu}_{wv}
        if that simultaneous-pair symmetry is the chosen convention.
    """
    return combine(expr)
