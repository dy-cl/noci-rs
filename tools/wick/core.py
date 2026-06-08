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
    """
    Orbital reference space.

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
    """
    Spin-free orbital index.

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
    """
    Spin-orbital fermion operator.

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
    """
    One generalized normal ordered group.

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
    """
    Product of GNO groups.

    Notation:
        \{E^p_q\}\{E^{rs}_{tu}\}

    Example:
        Product((tau1(u, v), tau1(x, w))) is \{E^u_v\}\{E^x_w\}
    """
    groups: tuple[Group, ...]

@dataclass(frozen = True, order = True)
class Delta:
    """
    Kronecker delta.

    Notation:
        \delta^p_q

    Example:
        Delta(i, j) is \delta^i_j
    """
    left: Idx
    right: Idx

@dataclass(frozen = True, order = True)
class Tensor:
    """
    Spin-free tensor factor.

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
    """
    One scalar product of deltas and tensors.

    Notation:
        c \delta ... \Gamma ... \Lambda ...

    Examples:
        Term(
            coeff = Fraction(1, 2),
            deltas = (Delta(i, j),),
            tensors = (Tensor("Gamma1", (u,), (w,)), Tensor("Theta", (x,), (v,))),
        )
        represents (1/2) \delta^i_j \Gamma^u_w \Theta^x_v
    """
    coeff: Fraction = Fraction(1)
    deltas: tuple[Delta, ...] = ()
    tensors: tuple[Tensor, ...] = ()

Expr = tuple[Term, ...]

def zero() -> Expr:
    """
    Return the zero expression.

    Notation:
        0

    Examples:
        A forbidden or vanishing contraction returns zero.
        \kappa(a^\dagger_{a\alpha}, a_{b\alpha}) = 0
        for virtual orbitals.
    """
    return ()

def one() -> Expr:
    """
    Return scalar one.

    Notation:
        1

    Examples:
        The empty product is one.
    """
    return (Term(),)

def delta(p: Idx, q: Idx) -> Expr:
    """
    Return a Kronecker delta expression.

    Notation:
        \delta^p_q

    Examples:
        delta(i, j) represents \delta^i_j.
        delta(a, b) represents \delta^a_b.
    """
    return (Term(deltas = (Delta(p, q),)),)

def tensor(name: str, upper: tuple[Idx, ...], lower: tuple[Idx, ...]) -> Expr:
    """
    Return one spin-free tensor expression.

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
    """
    Scale an expression.

    Notation:
        c A

    Examples:
        scale(tensor("Lambda2", (u, x), (v, w)), Fraction(-1, 2))
        represents -1/2 \Lambda^{ux}_{vw}
    """
    
    # Normalise input coefficient to a Fraction.
    c = coeff if isinstance(coeff, Fraction) else Fraction(coeff)
    
    # If the coefficient is zero the scaled expression is zero.
    if c == 0:
        return zero()

    return tuple(
        Term(
            # New coefficient after scaling.
            coeff = c * term.coeff,
            # Delta factors symbolic so unchanged.
            deltas = term.deltas,
            # Tensors symbolic so unchanged.
            tensors = term.tensors,
        )
        # Do this for every term in the expression.
        for term in expr
        # Provided the scaling does not give a zero.
        # If it does give a zero the scaled term is simply ignored.
        if c * term.coeff != 0
    )

def add(*exprs: Expr) -> Expr:
    """
    Add expressions and combine like terms.

    Notation:
        A + B + ...

    Examples:
        add(
            tensor("Gamma1", (u,), (v,)),
            tensor("Lambda2", (u, x), (v, w)),
        )
        represents \Gamma^u_v + \Lambda^{ux}_{vw}
    """

    # Collect all terms into list.
    terms = []

    for expr in exprs:
        # Add each expression to the list.
        terms.extend(expr)
    
    # Convert to tuple and simply with merging of terms.
    return combine(tuple(terms))

def mul(left: Expr, right: Expr) -> Expr:
    """
    Multiply scalar expressions commutatively.

    Notation:
        A B

    Examples:
        mul(delta(i, j), tensor("Gamma1", (u,), (w,)))
        represents \delta^i_j \Gamma^u_w
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
    
    # Covert to tuple and simplify with merging of terms.
    return combine(tuple(out))

def prod(exprs: tuple[Expr, ...]) -> Expr:
    """
    Multiply many scalar expressions by repeatedly multiplying.

    Notation:
        \prod A_k

    Examples:
        prod((
            delta(i, j),
            tensor("Gamma1", (u,), (w,)),
            tensor("Theta", (x,), (v,)),
        ))
        represents \delta^i_j \Gamma^u_w \Theta^x_v
    """
    # Begin with unity.
    out = one()
    
    # For each expression multiply it by the accumulated product.
    for expr in exprs:
        out = mul(out, expr)

    return out

def combine(expr: Expr) -> Expr:
    """
    Combine identical scalar products.

    Notation:
        c A + d A = (c + d) A

    Examples:
        (1/2) \Gamma^u_v + (1/2) \Gamma^u_v
        becomes \Gamma^u_v
    """
    # Accumulated coefficient dictionary with key (term.deltas, term.tensors) 
    # and value total accumulated coefficient thus far.
    acc: dict[tuple, Fraction] = {}

    # For every term in the expression form a key and add its coefficient 
    # to the running accumulation for that key. If the key is new acc.get
    # will correctly return zero.
    for term in expr:
        key = (term.deltas, term.tensors)
        acc[key] = acc.get(key, Fraction(0)) + term.coeff
    
    # Convert accumulator dictionary back into tuple of Term objects.
    return tuple(
        Term(
            coeff = coeff,
            deltas = deltas,
            tensors = tensors,
        )
        # Do it for each unique symbolic product and coefficient in dictionary.
        for (deltas, tensors), coeff in sorted(acc.items())
        # Provided that coefficient is not zero, otherwise ignore.
        if coeff != 0
    )

def anticommutator(left: Op, right: Op) -> Expr:
    """
    Canonical anticommutation relation.

    Notation:
        \{a_{p\sigma}, a^\dagger_{q\tau}\} = \delta^p_q \delta_{\sigma\tau}

    Examples:
        anticommutator(a_{p\alpha}, a^\dagger_{q\alpha}) gives \delta^p_q.

        anticommutator(a_{p\alpha}, a^\dagger_{q\beta}) gives 0.

        anticommutator(a^\dagger_{p\alpha}, a^\dagger_{q\alpha}) gives 0.

        anticommutator(a_{p\alpha}, a_{q\alpha}) gives 0.
    """
    # Only anticommute operators with left annihilator and right creator.
    if left.kind != "annihilate" or right.kind != "create":
        return zero()
    
    # If spins do not match the anticommutator is zero.
    if left.spin != right.spin:
        return zero()
    
    # If the orbital spaces do not match the anticommutator is zero.
    if left.idx.space != right.idx.space:
        return zero()
    
    # Otherwise the anticommutator becomes a delta.
    return delta(left.idx, right.idx)

def permutationSign(sequence: tuple[int, ...]) -> int:
    """
    Return fermionic parity of a permutation.

    Notation:
        Every swap of fermion operators contributes a factor of -1.

    Examples:
        permutationSign((1, 0)) gives -1 because a_p a_q = -a_q a_p.
        permutationSign((0, 2, 1, 3)) gives -1 because the sequence has one inversion.
    """
    inversions = 0

    for i in range(len(sequence)):
        # Compare each i with later positions j > i.
        for j in range(i + 1, len(sequence)):
            # If i is larger than any j pair is unordered so we have an inversion.
            if sequence[i] > sequence[j]:
                inversions += 1

    return -1 if inversions % 2 else 1

@cache
def partitions(items: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """
    Return set partitions of operator positions. Given a set of operator positions,
    return every possible way of grouping into blocks. 

    Notation:
        Generalised Wick theorem sums over partitions P:

            \langle \Phi | \{A\}\{B\}... | \Phi \rangle = sum_P sign(P) product_{B \in P} \langle \Phi | B | \Phi \rangle,

        where B is some block.

    Examples:
        For items = (0, 1, 2) the function will return:

            (0), (1), (2),
            (0), (1, 2),
            (0, 1), (2),
            (0, 2), (1),
            (0, 1, 2),

        each of which represents \langle \Phi | (0, 1) | \Phi \rangle \langle \Phi | (2) | \Phi \rangle for example. 
    """
    # If the set is empty the only partition is empty.
    if not items:
        return ((),)
    
    # Take first item out of tuple.
    first = items[0]
    rest = items[1:]
    out = []
    
    # Recursively get partitions of the smaller set.
    for partition in partitions(rest):

        # First option is to put first in its own block.
        out.append(((first,),) + partition)
        
        # Second option is to put first into each existing block one at a time.
        for i, block in enumerate(partition):
            out.append(
                partition[:i]
                + (tuple(sorted((first,) + block)),)
                + partition[i + 1:]
            )
    
    # Canonicalise parition list and remove duplicates.
    canonical = {
        tuple(sorted(partition, key = lambda block: block[0]))
        for partition in out
    }

    return tuple(sorted(canonical))

def partitionSign(partition: tuple[tuple[int, ...], ...]) -> int:
    """
    Return fermion sign for a Wick block pattern relative to increasing order.

    Notation:
        sign(P) is the parity of the flattened block order.

    Examples:
        partition = ((0, 2), (1, 3))

        Flattened order:

            (0, 2, 1, 3)

        There is one inversion relative to (0, 1, 2, 3), so sign(P) = -1.
    """
    sequence = tuple(
        i
        for block in partition
        for i in block
    )

    return permutationSign(sequence)

class Spin:
    """
    Spin-orbital to spin-free projection rules. This class converts spin-orbital 
    cumulants such as \gamma^{u_alpha}_{v_alpha} or \lambda^{u_alpha x_beta}_{v_alpha w_beta}
    into the spin free quantities \Gamma^u_v and \Lambda^{ux}_{vw}. The Wick expansion is performed 
    using explicit spin orbital fermionic operators, but the GNOCC equations are spin free.

    Notation:
        \gamma^{p_\sigma}_{q_\tau} ---> \Gamma^p_q

        \lambda^{p_\sigma q_\tau}_{r_\mu s_\nu} ---> \Lambda^{pq}_{rs}

        \lambda_3 ---> \Lambda_3

        \lambda_4 ---> \Lambda_4

    Examples:
        \gamma^{u_\alpha}_{v_\alpha} --> (1/2) \Gamma^u_v

        \lambda^{p_\alpha q_\beta}_{r_\alpha s_\beta} --> (1/6)(2 \Lambda^{pq}_{rs} + \Lambda^{pq}_{sr})
    """

    @staticmethod
    def projectCumulant(
        rank: int,
        upper: tuple[Idx, ...],
        lower: tuple[Idx, ...],
        upperSpins: tuple[str, ...],
        lowerSpins: tuple[str, ...],
    ) -> Expr:
        """
        Project one spin-orbital cumulant component.

        Notation:
            \gamma^{p_\sigma}_{q_\tau} ---> \Gamma^p_q
            \lambda^{p_\sigma q_\tau}_{r_\mu s_\nu} ---> \Lambda^{pq}_{rs}

        Examples:
            projectCumulant(1, (u,), (v,), ("alpha",), ("alpha",))
            represents gamma^{u_\alpha}_{v_\alpha} ---> (1/2) \Gamma^u_v

            projectCumulant(2, (u, x), (v, w), ("alpha", "beta"), ("alpha", "beta"))
            represents lambda^{u_\alpha x_\beta}_{v_\alpha w_\beta} ---> (1/6)(2 \Lambda^{ux}_{vw} + \Lambda^{ux}_{wv})

            projectCumulant(4, (p, r, t, v), (q, s, u, w), spins, spins)
            represents lambda^{p_{\sigma_1} r_{\sigma_2} t_{\sigma_3} v_{\sigma_4}}_{q_{\sigma_1} s_{\sigma_2} u_{\sigma_3} w_{\sigma_4}}
            ---> sum_{\pi in S_4} c_\pi \Lambda^{prtv}_{\pi(qsuw)}
        """

        # If tensor is rank 1 the spin free one-body density is just the spin sum.
        if rank == 1:
            return Spin.gamma1(
                upper,
                lower,
                upperSpins,
                lowerSpins,
            )
        
        # Map rank to cumulant name.
        names = {
            2: "Lambda2",
            3: "Lambda3",
            4: "Lambda4",
        }

        if rank not in names:
            raise NotImplementedError(f"rank {rank} active cumulant unsupported")
        
        # For ranks 2, 3, 4 we compute:
        # \lambda^{p_1 \sigma_1 \cdots p_k \sigma_k}_{q_1 \tau_1 \cdots q_k \tau_k} 
        # = \sum_{\pi} c_{\pi} \Lambda^{p_1 \cdots p_k}_{q_{\pi(1)} \cdots q_{\pi(k)}}
        # where \pi \in S_k is the sum over all permutations of k objects. See the function 
        # for more details.
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
        """
        Rank-1 spin summation.

        Notation:
            \Gamma^u_v = \gamma^{u_\alpha}_{v_\alpha} + \gamma^{u_\beta}_{v_\beta}

        Examples:
            gamma^{u_\alpha}_{v_\alpha} = (1/2) \Gamma^u_v
            gamma^{u_\alpha}_{v_\beta} = 0
        """

        # If upper and lowe spin labels differ the spin-free object is zero.
        if upperSpins[0] != lowerSpins[0]:
            return zero()
        
        # Equal-spin components are half of the spin free density.
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
        """
        Rank-k spin summation.

        Notation:
            \lambda^{p_1 \sigma_1 \cdots p_k \sigma_k}_{q_1 \tau_1 ... q_k \tau_k}
            = \sum_{\pi in S_k} c_\pi \Lambda^{p_1 \cdots p_k}_{\pi(q_1 \cdots q_k)}

        Examples:
            \lambda^{p_\alpha q_\alpha}_{r_\alpha s_\alpha}
            = (1/6)(\Lambda^{pq}_{rs} - \Lambda^{pq}_{sr})

            \lambda^{p_\alpha q_\beta}_{r_\alpha s_\beta}
            = (1/6)(2 \Lambda^{pq}_{rs} + \Lambda^{pq}_{sr})

            lambda^{t_\alpha y_\beta z_\alpha}_{u_\alpha w_\beta x_\alpha}
            = \sum_{\pi in S_3} c_\pi \Lambda^{tyz}_{\pi(uwx)}

            lambda^{p_\alpha r_\beta t_\alpha v_\beta}_{q_\alpha s_\beta u_\alpha w_\beta}
            = \sum_{\pi in S_4} c_\pi \Lambda^{prtv}_{\pi(qsuw)}
        """
        # Begin with zero expression.
        out = zero()
        
        # Loop over every non-zero permutation coefficient c_\pi of the lower spin indices.
        for lowerPerm, coeff in Spin.coeffs(
            len(upper),
            upperSpins,
            lowerSpins,
        ):
            # Add to previous elements of the sum with scale c_\pi.
            out = add(
                out,
                scale(
                    # Construct the tensor \Lambda^{p_1 \cdots p_k}_{q_{\pi(1)} \cdots q_{\pi(k)}}
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
        """
        Return lower-permutation spin-replacement coefficients c_\pi required for spin summation of cumulants.

        Notation:
            \lambda^{p_1 \sigma_1 ... p_k \sigma_k}_{q_1 \tau_1 ... q_k \tau_k}
            = sum_{\pi in S_k} c_\pi Lambda^{p_1...p_k}_{q_{\pi(1)}...q_{\pi(k)}}

        Examples:
            lambda^{p_\alpha q_\beta}_{r_\alpha s_\beta}
            = (1/6)(2 Lambda^{pq}_{rs} + Lambda^{pq}_{sr})

            lambda^{p_\sigma q_\tau r_\mu}_{s_\nu t_\rho u_\eta}
            = sum_{\pi in S_3} c_\pi Lambda^{pqr}_{\pi(stu)}
            up to a gauge convention.

        Note that for rank 3 and above the lower-permutation spin-free basis is overcomplete because the spin 
        space has only two spin functions. This makes the coefficients c_\pi not unique due to the presence of a 
        null-space. A gauge is fixed in solveConsistent() by setting free variables to zero to give a deterministic 
        outcome.
        """
        if rank not in (2, 3, 4):
            raise NotImplementedError(f"Rank {rank} spin replacement coefficients are not implemented.")

        if len(upperSpins) != rank or len(lowerSpins) != rank:
            raise ValueError("Spin tuples do not match cumulant rank.")
        
        # Get all possible permutations of the lower spin indices.
        perms = tuple(permutations(range(rank)))

        def invPerm(p: tuple[int, ...]) -> tuple[int, ...]:
            """
            Return the inverse of a permutation. 

            Notation:
                p maps i ---> p[i], invPerm(p) maps p[i] ---> i.

            Examples:
                p = (2, 0, 1) means that:
                0 ---> 2,
                1 ---> 0,
                2 ---> 1,
                and therefore invPerm(p) = (1, 2, 0).
            """
            out = [0] * len(p)

            for i, x in enumerate(p):
                out[x] = i

            return tuple(out)

        def compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
            """
            Return the composition of two permutation maps.

            Notation:
                p maps i ---> p[i],
                q maps i ---> q[i],
                compose(p, q) represents i ---> q[i] ---> p[q[i]].

            Examples:
                p = (2, 0, 1) means that:
                0 ---> 2,
                1 ---> 0,
                2 ---> 1.

                q = (1, 2, 0) means that:
                0 ---> 1,
                1 ---> 2,
                2 ---> 0.

                Then compose(p, q) = p after q.

                For i = 0:
                0 ---> q[0] = 1 ---> p[1] = 0.
                For i = 1:
                1 ---> q[1] = 2 ---> p[2] = 1.
                For i = 2:
                2 ---> q[2] = 0 ---> p[0] = 2.

                Therefore:
                compose(p, q) = (0, 1, 2).

            """
            return tuple(p[q[i]] for i in range(len(p)))

        def cycleCount(p: tuple[int, ...]) -> int:
            """
            Count the number of cycles in a permutation. A cycle is a closed chain within a permutation.
            
            Notation:
                p maps i ---> p[i].
                Starting from an index i apply p repeatedly.
                i ---> p[i] ---> p[p[i]] ---> ...
                This cycle will eventually return to an index visited previously.
                This closed chain is one cycle.

            Examples:
                p = (1, 2, 0) means:
                0 ---> 1,
                1 ---> 2, 
                2 ---> 0,

                If we start from 0: 0 ---> 1 ---> 2 ---> 0 so all indices are in the same cycle.
                Therefore cycleCount((1, 2, 0)) = 1.
                Conversley the identity permutation p = (0, 1, 2) gives cycleCount = 3.
            """
            # Track seen indices within a cycle.
            seen = [False] * len(p)
            cycles = 0
            
            # Start a cycle from all indices.
            for i in range(len(p)):
                # If this index was visited in another cycle already skip it.
                if seen[i]:
                    continue
                
                # New unvisited index starts a new cycle.
                cycles += 1
                j = i
                
                # Follow permutation map until returning to a visited index.
                while not seen[j]:
                    seen[j] = True
                    j = p[j]

            return cycles

        def solveConsistent(mat: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
            """
            Solve a linear system A x = b that may be singular and choose a deterministic gauge. 

            Here, A is the spin-projection Gram matrix and x are the coefficients c_\pi in:
            \lambda^{p_1 \sigma_1 ... p_k \sigma_k}_{q_1 \tau_1 ... q_k \tau_k}
            = sum_{\pi in S_k} c_\pi Lambda^{p_1...p_k}_{q_{\pi(1)}...q_{\pi(k)}}.

            This system can be singular because the spin-free \Lambda permutation basis 
            is linearly dependent for rank 3 or greater. This function chooses a deterministic solve 
            by setting all free variables to zero.
            """
            # Number of equations.
            nRows = len(mat)
            # Number of unknowns (number of lower-permutations).
            nCols = len(mat[0])
            # Combine A and b into one matrix for Gaussian elimination.
            aug = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
            
            # Record pivot columns, those which correspond to determined variables.
            pivotCols = []
            row = 0
            
            # Scan columns left to right and try to find a pivot in each.
            for col in range(nCols):
                pivot = None
                
                # First entry in column j with r >= row where A_{rj} \neq 0 becomes a pivot.
                for r in range(row, nRows):
                    if aug[r][col] != 0:
                        pivot = r
                        break
                
                # If we have no pivot column j corresponds to a free variable.
                if pivot is None:
                    continue
                
                # Swap pivot and row.
                if pivot != row:
                    aug[row], aug[pivot] = aug[pivot], aug[row]
                
                # Normalise pivot row.
                scaleFactor = aug[row][col]
                aug[row] = [x / scaleFactor for x in aug[row]]
                
                # Eliminate the pivot column from all other rows (Gauss-Jordan elimination).
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
                
                # Now all rows have pivots.
                if row == nRows:
                    break

            for r in range(row, nRows):
                if all(aug[r][c] == 0 for c in range(nCols)) and aug[r][-1] != 0:
                    raise ValueError("inconsistent spin-projection system")
            
            # Initialise solution with all variables zero.
            sol = [Fraction(0) for _ in range(nCols)]
            
            # Insert pivot variables into solution.
            for r, col in enumerate(pivotCols):
                sol[col] = aug[r][-1]

            return sol
        
        # Setup spin-projection linear system by first building Gram matrix.
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
        
        # Setup spin-projection linear system by building RHS vector.
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
        
        # Get solution.
        sol = solveConsistent(gram, rhs)

        return tuple(
            (p, c)
            for p, c in zip(perms, sol)
            if c != 0
        )

class Ref:
    """
    Reference-state connected cumulant rules. 

    Notation:
        \kappa(op_1, op_2, \cdots) is representative of is the connected reference cumulant
        of the operator block.

    Examples:
        \kappa(a^\dagger_{i\alpha}, a_{j\alpha}) = \delta^i_j
        \kappa(a_{a\alpha}, a^\dagger_{b\alpha}) = \delta^a_b
    """

    def kappa(self, ops: tuple[Op, ...]) -> Expr:
        """
        Connected cumulant of one Wick block.

        Notation:
            \kappa(B) = \langle \Phi | B | \Phi \rangle.

        Examples:
            \kappa(a^\dagger_{i\alpha}, a_{j\alpha})
            = \delta^i_j

            \kappa(a_{a\alpha}, a^\dagger_{b\alpha})
            = \delta^a_b

            \kappa(a^\dagger_{u\alpha}, a_{v\alpha})
            = \gamma^{u_\alpha}_{v_\alpha} ---> (1/2) \Gamma^u_v

            \kappa(a^\dagger_{u\alpha}, a^\dagger_{x\beta}, a_{w\beta}, a_{v\alpha})
            = lambda^{u_\alpha x_\beta}_{v_\alpha w_\beta} ---> spin-free \Lambda_2 combination
        """

        # If there's no operators return zero.
        if not ops:
            return zero()
        
        # Get the orbital spaces in this cumulant block.
        spaces = {op.idx.space for op in ops}
        
        # Go to active cumulant rules.
        if spaces == {Space.ACTIVE}:
            return self.activeKappa(ops)
        
        # If the block is not active only the only non-zero term are two operator 
        # contractions in core or virtual spaces.
        if len(ops) == 2:
            return self.frozenPair(ops[0], ops[1])

        return zero()

    def frozenPair(self, left: Op, right: Op) -> Expr:
        """
        Core/virtual deterministic cumulants.

        Notation:
            \kappa(a^\dagger_{i\sigma}, a_{j\tau})
            = \delta^i_j \delta_{\sigma\tau}

            \kappa(a_{a\sigma}, a^\dagger_{b\tau})
            = \delta^a_b \delta_{\sigma\tau}

        Examples:
            frozenPair(a^\dagger_{i\alpha}, a_{j\alpha})
            = \delta^i_j

            frozenPair(a_{a\alpha}, a^\dagger_{b\alpha})
            = \delta^a_b

            frozenPair(a_{i\alpha}, a^\dagger_{j\alpha})
            = 0

            frozenPair(a^\dagger_{a\alpha}, a_{b\alpha})
            = 0
        """
        # Conserve spin.
        if left.spin != right.spin:
            return zero()
        
        # Ensure orbital indices belond to the same space.
        if left.idx.space != right.idx.space:
            return zero()
        
        # If both indices are core we have:
        # \kappa(a^\dagger_i, a_j) = \langle \Phi | a^\dagger_i a_j | \Phi \rangle = \delta_i^j
        if left.idx.space == Space.CORE:
            if left.kind == "create" and right.kind == "annihilate":
                return delta(left.idx, right.idx)
            
            # Opposite ordering is the core-hole contraction and vanishes because
            # core orbitals are assumed fully occupied.
            return zero()
        
        # If both indices are virtual then we have:
        # \kappa(a_a, a^\dagger_b) = \delta^a_b
        if left.idx.space == Space.VIRTUAL:
            if left.kind == "annihilate" and right.kind == "create":
                return delta(left.idx, right.idx)
            
            # Opposite ordering vanishes because virtual orbitals are assumed empty.
            return zero()

        return zero()

    def activeKappa(self, ops: tuple[Op, ...]) -> Expr:
        """
        Evaluate a connected cumulant block containing only active-space operators.

        Notation:
            \kappa(a^\dagger_{u\sigma}, a_{v\sigma})
            =
            \gamma^{u_\sigma}_{v_\sigma}
            ->
            (1/2) \Gamma^u_v

            \kappa(a_{u\sigma}, a^\dagger_{v\sigma})
            =
            \eta^{v_\sigma}_{u_\sigma}
            ->
            (1/2) \Theta^v_u

            \kappa(
                a^\dagger_{p_1\sigma_1}
                ...
                a^\dagger_{p_k\sigma_k}
                a_{q_k\tau_k}
                ...
                a_{q_1\tau_1}
            )
            =
            \lambda^{p_1\sigma_1 ... p_k\sigma_k}_{q_1\tau_1 ... q_k\tau_k}
            ->
            \sum_{\pi \in S_k}
                c_\pi
                \Lambda^{p_1...p_k}_{q_{\pi(1)}...q_{\pi(k)}}

        Examples:
            kappa(a^\dagger_{u\alpha}, a_{v\alpha})
            =
            gamma^{u_\alpha}_{v_\alpha}
            ->
            (1/2) \Gamma^u_v

            kappa(a_{u\alpha}, a^\dagger_{v\alpha})
            =
            eta^{v_\alpha}_{u\alpha}
            ->
            (1/2) \Theta^v_u

            kappa(a^\dagger_{u\alpha}, a^\dagger_{x\beta}, a_{w\beta}, a_{v\alpha})
            =
            lambda^{u_\alpha x_\beta}_{v_\alpha w_\beta}
            ->
            (1/6)(2 \Lambda^{ux}_{vw} + \Lambda^{ux}_{wv})

            kappa(a^\dagger_{t\alpha}, a^\dagger_{y\beta}, a^\dagger_{z\alpha},
                  a_{x\alpha}, a_{w\beta}, a_{u\alpha})
            =
            lambda^{t_\alpha y_\beta z_\alpha}_{u_\alpha w_\beta x_\alpha}
            ->
            \sum_{\pi \in S_3}
                c_\pi
                \Lambda^{tyz}_{\pi(uwx)}
        """
        # Handle active one-body case specifically.
        if len(ops) == 2:
            left, right = ops
            
            # Spin conservation.
            if left.spin != right.spin:
                return zero()
            
            # \kappa(a^\dagger_{u sigma}, a_{v \sigma})
            # = \gamma^{u_\sigma}_{v_\sigma} = (1/2) \Gamma^u_v.
            if left.kind == "create" and right.kind == "annihilate":
                return scale(
                    tensor("Gamma1", (left.idx,), (right.idx,)),
                    Fraction(1, 2),
                )
            
            # \kappa(a_{u \sigma}, a^\dagger_{v \sigma})
            # = \eta^{v_\sigma}_{u_\sigma}
            # = (1/2) \Theta^v_u
            if left.kind == "annihilate" and right.kind == "create":
                return scale(
                    tensor("Theta", (right.idx,), (left.idx,)),
                    Fraction(1, 2),
                )
            
            # Creation-creation or annihilation-annhilation vanishes.
            return zero()
        
        # For higher active space cumulants first normal order the operator block.
        normal = self.normalOrder(ops)

        if normal is None:
            return zero()

        sign, creators, annihilators = normal
        rank = len(creators)
        
        # Make sure we are number conserving.
        if rank != len(annihilators):
            return zero()
        
        # Get upper indices and their spins.
        upper = tuple(op.idx for op in creators)
        upperSpins = tuple(op.spin for op in creators)
        
        # Get lower indices and their spins.
        # Reversed to maintain convention.
        lower = tuple(op.idx for op in reversed(annihilators))
        lowerSpins = tuple(op.spin for op in reversed(annihilators))
        
        # Project spin-orbital cumulant into spin-free space.
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
        """
        Bring active operators to creator-left (normal) order and return fermionic sign of that ordering.
        Note that this function does not generate the anticommutator delta terms.

        Notation:
            a_{u\alpha} a^\dagger_{v\alpha}
            = \delta^u_v - a^\dagger_{v\alpha} a_{u\alpha}

         Example:
            a_u a^\dagger_v = - a^\dagger_v a_u
            along with some delta term.
        """
        # Split into creators and annihilators.
        creators = tuple(op for op in ops if op.kind == "create")
        annihilators = tuple(op for op in ops if op.kind == "annihilate")
        
        # Check consistency.
        if len(creators) + len(annihilators) != len(ops):
            return None
        
        # Count number of inversions for normal ordering.
        inversions = 0
        for i, left in enumerate(ops):
            if left.kind != "annihilate":
                continue

            for right in ops[i + 1:]:
                if right.kind == "create":
                    inversions += 1
        
        # Fermionic sign.
        sign = -1 if inversions % 2 else 1

        return sign, creators, annihilators

class Wick:
    """
    Generalised Wick evaluator. This class evaluates expectation value of 
    products of generalised normal-ordered spin-free excitation groups with 
    respect to the reference vacuum.

    Notation:
        \langle \Phi | \{A\}\{B\}\{C\} \cdots | \Phi \rangle.
        = \sum_{P} sign(P) \prod_{B \in P} \kappa(B).

    Examples:
        Wick(ref).eval(Product((tau1(i, u, 0), tau1(x, a, 1))))
        represents \langle \Phi | \{E^i_u\}\{E^x_a\} | \Phi \rangle.
    """

    def __init__(self, ref: Ref):
        self.ref = ref

    def eval(self, product: Product) -> Expr:
        """
        Evaluate a product of normal ordered groups.

        Notation:
            \langle \Phi | \{A\}\{B\}\{C\} \cdots | \Phi \rangle.
            = \sum_{P} sign(P) \prod_{B \in P} \kappa(B).

        Examples:
            product = Product((tau2(i, u, v, a, 0), tau2(x, b, j, w, 1)))
            represents \langle \Phi | \{E^{iu}_{va}\}\{E^{xb}_{jw}\} | \Phi \rangle

            The algorithm expands spin-free E operators into spin-orbital
            strings, enumerates Wick partitions, rejects internal contractions,
            applies Ref.kappa to each block, and sums the result.
        """

        # Accumulator.
        out = zero()
        
        # Expand spin-free groups into spin-orbital strings.
        for ops in self.expand(product):
            out = add(
                out,
                self.evalSpinString(ops),
            )
        
        # Simplify terms.
        return simplify(out)

    def expand(self, product: Product) -> tuple[tuple[Op, ...], ...]:
        """
        Expand spin-free groups into spin-orbital strings.

        Notation:
            \{E^u_v\}
            = \{a^\dagger_{u\alpha}a_{v\alpha}\} + \{a^\dagger_{u\beta}a_{v\beta}\}

            \{E^{ua}_{iv}\}
            = \sum_{\sigma\tau} \{a^\dagger_{u\sigma}a^\dagger_{a\tau}a_{v\tau}a_{i\sigma}\}

        Examples:
            Product((tau1(u, v, 0), tau1(x, w, 1))) expands into four spin strings.
        """
        strings = ((),)
        
        # Process normal ordered groups \hat E from left to right,
        for group in product.groups:
            nextStrings = []
            
            # Existing partial strings.
            for prefix in strings:
                # Next group's spin components.
                for suffix in group.strings:
                    nextStrings.append(prefix + suffix)
            
            # Update list of spin-strings
            strings = tuple(nextStrings)

        return strings

    def evalSpinString(self, ops: tuple[Op, ...]) -> Expr:
        """
        Evaluate one spin-orbital string by generalised Wick's theorem. A parition is rejected 
        if any block is internal to one normal-ordered group due to the normal ordering 
        or if Ref.kappa(block) is zero.

        Notation:
            \sum_P \sign(P) \prod_{B \in P} \kappa(B)

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
        # Operator positions labelled by ints.
        positions = tuple(range(len(ops)))
        
        # Enumerate every set partition of operator positions.
        for partition in partitions(positions):

            # Evaluated cumulant factors.
            factors = []
            valid = True
            
            # For every block in parition, reject internal contractions,
            # evaluate connected cumulant and keep non-zero block factor.
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
            
            # Contribution from a given parition is \sign(P) \prod_B \kappa(B).
            out = add(
                out,
                scale(
                    prod(tuple(factors)),
                    self.sign(partition),
                ),
            )

        return out

    def internal(self, ops: tuple[Op, ...], block: tuple[int, ...]) -> bool:
        """
        Check whether a Wick block is internal to one group as internal 
        contractions inside \{...\} are forbidden. A block is internal if 
        every operator in the block has the same group id.

        Examples:
            In \{a^\dagger_{u\alpha}a_{v\alpha}\} the block
            (a^\dagger_{u\alpha}, a_{v\alpha}) is internal and therefore rejected.

            In \{a^\dagger_{u\alpha}a_{v\alpha}\} \{a^\dagger_{x\alpha}a_{w\alpha}\}
            the block (a_{v\alpha}, a^\dagger_{x\alpha}) connects two groups and therefore allowed.
        """
        groups = {
            ops[i].group
            for i in block
        }

        return len(groups) == 1

    def sign(self, partition: tuple[tuple[int, ...], ...]) -> int:
        """
        Fermion sign for a Wick partition. Return parity of a block 
        relative to the original increasing operator order.

        Notation:
            sign(P) is the parity of the flattened block ordering.

        Examples:
            partition = ((0, 2), (1, 3)) gives a flattened sequence:

                (0, 2, 1, 3)

            which has one inversion from (0, 1, 2, 3), so sign(P) = -1.
        """
        return partitionSign(partition)

def groupE1(p: Idx, q: Idx, groupId: int) -> Group:
    """
    Build spin components of spin-free E^p_q.

    Notation:
        E^p_q = \sum_\sigma a^\dagger_{p\sigma} a_{q\sigma}

    Examples:
        groupE1(u, v, 0) represents
        E^u_v = a^\dagger_{u\alpha}a_{v\alpha} + a^\dagger_{u\beta}a_{v\beta}

        If used in a Product as a GNO group, this is \{E^u_v\}.
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
    """
    Build spin components of spin-free E^{pq}_{rs}.

    Notation:
        E^{pq}_{rs} = \sum_{\sigma\tau} a^\dagger_{p\sigma} a^\dagger_{q\tau} a_{s\tau} a_{r\sigma}

    Examples:
        groupE2(u, a, i, v, 0) represents
        E^{ua}_{iv} = \sum_{\sigma\tau} a^\dagger_{u\sigma} a^\dagger_{a\tau} a_{v\tau} a_{i\sigma}

        If used in a Product as a GNO group, this is \{E^{ua}_{iv}\}.
    """
    strings = []

    for sigma in SPINS:
        for tau in SPINS:
            ops = (
                Op("create", p, sigma, groupId),
                Op("create", q, tau, groupId),
                Op("annihilate", s, tau, groupId),
                Op("annihilate", r, sigma, groupId),
            )

            if len(set(ops)) == len(ops):
                strings.append(ops)

    return Group(tuple(strings))

def tau1(create: Idx, annihilate: Idx, groupId: int) -> Group:
    """
    One-body GNO excitation.

    Notation:
        \{E^p_q\}

    Examples:
        tau1(u, i, 0) represents \{E^u_i\}.
    """
    return groupE1(create, annihilate, groupId)

def tau2(
    create1: Idx,
    create2: Idx,
    annihilate1: Idx,
    annihilate2: Idx,
    groupId: int,
) -> Group:
    """
    Two-body GNO excitation.

    Notation:
        \{E^{pq}_{rs}\}

    Examples:
        tau2(u, a, i, v, 0) represents \{E^{ua}_{iv}\}.
    """
    return groupE2(
        create1,
        create2,
        annihilate1,
        annihilate2,
        groupId,
    )

def daggerTau1(create: Idx, annihilate: Idx, groupId: int) -> Group:
    """
    Adjoint of a one-body excitation.

    Notation:
        (\{E^p_q\})^\dagger = \{E^q_p\}

    Examples:
        daggerTau1(u, i, 0) represents (\{E^u_i\})^\dagger = \{E^i_u\}.
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
    """
    Adjoint of a two-body excitation.

    Notation:
        (\{E^{pq}_{rs}\})^\dagger = \{E^{rs}_{pq}\}

    Examples:
        daggerTau2(u, a, i, v, 0) represents (\{E^{ua}_{iv}\})^\dagger = \{E^{iv}_{ua}\}.
    """
    return tau2(
        annihilate1,
        annihilate2,
        create1,
        create2,
        groupId,
    )

def simplify(expr: Expr) -> Expr:
    """
    Algebraic simplification only. Currently just a wrapper to combine.
    Candidate for deletion.
    """
    return combine(expr)
