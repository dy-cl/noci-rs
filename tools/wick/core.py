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

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from itertools import combinations, permutations, product as cartesianProduct
import os
from time import perf_counter

try:
    from gmpy2 import mpq as Fraction
except ImportError:
    from fractions import Fraction

from spinsum import permutationSign, spinGram, solveConsistent

ZERO_FRACTION = Fraction(0)

# Large spin-string results caused OOM by staying in Wick.spinStringCache.
# Keep caching useful small results, but never retain pathological expressions.
SPINSTRING_CACHE_TERM_LIMIT = 100_000

# Periodically rebuild huge coefficient tables to shed hash-table slack. The
# value is high enough to avoid slowing ordinary classes, low enough to bound
# pathological A/A residual work items.
SPINSTRING_ACC_COMPACT_LIMIT = 200_000

# Stream only expressions above current Wick products. Existing 16-operator R2
# strings rely on cached partition suffixes for speed; streaming is reserved
# for future larger strings.
SPINSTRING_STREAM_OP_LIMIT = 18
SPINSTRING_STREAM_SUFFIX_CACHE_OP_LIMIT = 12

# Wick partition products are usually small. Use the old fast partial-product
# list until it would become too large, then stream the rest into the
# accumulator. This bounds peak memory without making ordinary products slow.
PRODUCT_PARTIAL_TERM_LIMIT = 10_000_000
WICK_PROGRESS_PARTITION_INTERVAL = 50_000
WICK_SPIN_POOL = None
WICK_SPIN_POOL_JOBS = 0

PROFILE_ENABLED = False
PROFILE_TIMES: dict[str, float] = {}
PROFILE_COUNTS: dict[str, int] = {}

@cache
def addCoeffValues(left: Fraction, right: Fraction) -> Fraction:
    """
    Add two exact scalar coefficients with a cache.

    Notation:
        c = a + b

    Examples:
    """
    return left + right

@cache
def mulCoeffValues(left: Fraction, right: Fraction) -> Fraction:
    """
    Multiply two exact scalar coefficients with a cache.

    Notation:
        c = a b

    Examples:
    """
    return left * right

def addCoeffValuesFast(left: Fraction, right: Fraction) -> Fraction:
    """
    Add two exact scalar coefficients with zero fast paths.

    Notation:
        c = a + b

    Examples:
        addCoeffValuesFast(0, c) returns c without constructing a new rational.
    """
    if left == 0:
        return right

    if right == 0:
        return left

    return addCoeffValues(
        left,
        right,
    )

def mulCoeffValuesFast(left: Fraction, right: Fraction) -> Fraction:
    """
    Multiply two exact scalar coefficients with zero and sign fast paths.

    Notation:
        c = a b

    Examples:
        mulCoeffValuesFast(-1, c) returns -c without rational multiplication.
    """
    if left == 1:
        return right

    if right == 1:
        return left

    if left == -1:
        return -right

    if right == -1:
        return -left

    if left == 0 or right == 0:
        return ZERO_FRACTION

    return mulCoeffValues(
        left,
        right,
    )

def setProfiling(enabled: bool) -> None:
    """
    Enable or disable low-overhead Wick timing counters.

    Notation:
        

    Examples:
        terms.py calls this when WICK_PROFILE=1 or --profile is used.
    """
    global PROFILE_ENABLED
    PROFILE_ENABLED = enabled

def resetProfile() -> None:
    """
    Clear Wick timing counters.

    Notation:
        

    Examples:
        Called before each residual class so timings stay class-local.
    """
    PROFILE_TIMES.clear()
    PROFILE_COUNTS.clear()

def profileSnapshot() -> tuple[dict[str, float], dict[str, int]]:
    """
    Return a copy of Wick timing counters.

    Notation:
        

    Examples:
        The emitter prints this on stderr after one class.
    """
    return dict(PROFILE_TIMES), dict(PROFILE_COUNTS)

@contextmanager
def timed(name: str):
    """
    Time one Wick kernel if profiling is enabled.

    Notation:
        

    Examples:
        with timed("Wick.expand"): ...
    """
    if not PROFILE_ENABLED:
        yield
        return

    start = perf_counter()
    try:
        yield
    finally:
        PROFILE_TIMES[name] = PROFILE_TIMES.get(name, 0.0) + perf_counter() - start
        PROFILE_COUNTS[name] = PROFILE_COUNTS.get(name, 0) + 1

def currentRssKiB() -> int:
    """
    Return the current resident set size in KiB.

    Notation:
        RSS is the resident memory currently held by the process.

    Examples:
        Wick progress prints currentRssKiB() during long spin-string partition loops.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0

def wickEnvInt(name: str, default: int) -> int:
    """
    Read one integer Wick runtime environment option.

    Notation:
        x = \mathrm{env}(name)

    Examples:
        wickEnvInt("WICK_SPIN_JOBS", 1) returns the requested spin workers.
    """
    value = os.environ.get(name)
    if value in (None, ""):
        return default

    try:
        return int(value)
    except ValueError:
        return default

def wickEnvPositiveInt(name: str, default: int) -> int:
    """
    Read one positive integer Wick runtime environment option.

    Notation:
        x = \max(1, \mathrm{env}(name))

    Examples:
        WICK_SPIN_CHUNK_SIZE=1 schedules one spin string per worker task.
    """
    return max(
        1,
        wickEnvInt(
            name,
            default,
        ),
    )

def wickEnvRange(
    startName: str,
    stopName: str,
    total: int,
) -> tuple[int, int, bool]:
    """
    Return a 1-based stop-exclusive environment slice as 0-based bounds.

    Notation:
        [s, t) \subseteq \{1,\ldots,N\}

    Examples:
        WICK_SPIN_START=70 and WICK_SPIN_STOP=90 selects spin strings
        70 through 89.
    """
    startValue = os.environ.get(startName)
    stopValue = os.environ.get(stopName)
    partial = startValue not in (None, "") or stopValue not in (None, "")

    start = wickEnvInt(startName, 1)
    stop = wickEnvInt(stopName, total + 1)

    start = max(1, start)
    stop = min(total + 1, max(start, stop))

    return (
        start - 1,
        stop - 1,
        partial,
    )

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

@dataclass(frozen = True, order = True, slots = True)
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
    _hash: int = field(init = False, repr = False, compare = False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash((self.name, self.space.value)),
        )

    def __hash__(self) -> int:
        return self._hash

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

@dataclass(frozen = True, order = True, slots = True)
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
    _hash: int = field(init = False, repr = False, compare = False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash((self.left, self.right)),
        )

    def __hash__(self) -> int:
        return self._hash

@dataclass(frozen = True, order = True, slots = True)
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
    _hash: int = field(init = False, repr = False, compare = False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash((self.name, self.upper, self.lower)),
        )

    def __hash__(self) -> int:
        return self._hash

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
IntKey = tuple[tuple[int, ...], tuple[int, ...]]
IntAcc = dict[IntKey, Fraction]

class FactorInterner:
    """
    Map symbolic Delta/Tensor factors to compact integer ids.

    Notation:
        \iota: \mathcal{F} \to \mathbb{N}^{+}

    Examples:
        Parallel Wick workers return tensor ids instead of repeated Tensor
        objects in every accumulator key.
    """
    def __init__(self) -> None:
        self.deltaIds: dict[Delta, int] = {}
        self.tensorIds: dict[Tensor, int] = {}
        self.deltas: list[Delta] = []
        self.tensors: list[Tensor] = []

    def deltaId(self, deltaIn: Delta) -> int:
        """
        Return the id for one delta factor.

        Notation:
            \iota_{\Delta}(\delta^p_q) = n

        Examples:
            Equal Delta values get the same positive integer id.
        """
        value = self.deltaIds.get(deltaIn)
        if value is not None:
            return value

        value = len(self.deltas) + 1
        self.deltaIds[deltaIn] = value
        self.deltas.append(deltaIn)
        return value

    def tensorId(self, tensorIn: Tensor) -> int:
        """
        Return the id for one tensor factor.

        Notation:
            \iota_T(T^{p}_{q}) = n

        Examples:
            Equal Tensor values get the same positive integer id.
        """
        value = self.tensorIds.get(tensorIn)
        if value is not None:
            return value

        value = len(self.tensors) + 1
        self.tensorIds[tensorIn] = value
        self.tensors.append(tensorIn)
        return value

def encodeExprToIntAcc(interner: FactorInterner, expr: Expr) -> IntAcc:
    """
    Encode a symbolic expression into an integer-keyed accumulator.

    Notation:
        E = \sum_A c_A A \mapsto a_{\iota(A)} = c_A

    Examples:
        Spin workers use this before sending results back to the parent.
    """
    acc: IntAcc = {}

    for term in expr:
        key = (
            tuple(interner.deltaId(deltaIn) for deltaIn in term.deltas),
            tuple(interner.tensorId(tensorIn) for tensorIn in term.tensors),
        )
        old = acc.get(key)
        acc[key] = term.coeff if old is None else addCoeffValuesFast(old, term.coeff)

    return {
        key: coeff
        for key, coeff in acc.items()
        if coeff != 0
    }

def decodeIntAcc(interner: FactorInterner, acc: IntAcc, sort: bool = False) -> Expr:
    """
    Decode an integer-keyed accumulator into symbolic terms.

    Notation:
        a_{\iota(A)} \mapsto \sum_A a_{\iota(A)} A

    Examples:
        Wick.eval decodes once after merging parallel spin shards.
    """
    items = sorted(acc.items()) if sort else acc.items()
    out = []

    for (deltaIds, tensorIds), coeff in items:
        if coeff == 0:
            continue

        out.append(
            Term(
                coeff = coeff,
                deltas = tuple(interner.deltas[i - 1] for i in deltaIds),
                tensors = tuple(interner.tensors[i - 1] for i in tensorIds),
            )
        )

    return tuple(out)

def mergeIntShard(
    targetInterner: FactorInterner,
    targetAcc: IntAcc,
    shardDeltas: tuple[Delta, ...],
    shardTensors: tuple[Tensor, ...],
    shardAcc: IntAcc,
) -> None:
    """
    Merge one worker-local integer accumulator into a parent accumulator.

    Notation:
        a_{\iota(A)} \leftarrow a_{\iota(A)} + b_{\iota_w(A)}

    Examples:
        Parallel spin-string chunks are merged deterministically by spin index.
    """
    deltaMap = tuple(targetInterner.deltaId(deltaIn) for deltaIn in shardDeltas)
    tensorMap = tuple(targetInterner.tensorId(tensorIn) for tensorIn in shardTensors)

    for (deltaIds, tensorIds), coeff in shardAcc.items():
        key = (
            tuple(deltaMap[i - 1] for i in deltaIds),
            tuple(tensorMap[i - 1] for i in tensorIds),
        )
        old = targetAcc.get(key)

        if old is None:
            targetAcc[key] = coeff
            continue

        value = addCoeffValuesFast(
            old,
            coeff,
        )
        if value == 0:
            del targetAcc[key]
        else:
            targetAcc[key] = value

def wickSpinPool(jobs: int):
    """
    Return the lazy process pool used for spin-string parallelism.

    Notation:
        \mathcal{P}_{n} = \mathrm{ProcessPool}(n)

    Examples:
        WICK_SPIN_JOBS=16 creates one persistent pool for Wick.eval calls.
    """
    global WICK_SPIN_POOL
    global WICK_SPIN_POOL_JOBS

    if WICK_SPIN_POOL is None or WICK_SPIN_POOL_JOBS != jobs:
        if WICK_SPIN_POOL is not None:
            WICK_SPIN_POOL.shutdown()

        WICK_SPIN_POOL = ProcessPoolExecutor(max_workers = jobs)
        WICK_SPIN_POOL_JOBS = jobs

    return WICK_SPIN_POOL

def _wickSpinChunkWorker(args):
    """
    Evaluate a chunk of spin strings in a worker process.

    Notation:
        S = \sum_{s \in C} W_s

    Examples:
        Wick.eval submits one or more spin strings and receives one compact
        integer accumulator shard.
    """
    (
        chunkStart,
        spinChunk,
        connected,
        maxActiveCumulantRank,
    ) = args
    setProfiling(False)
    os.environ["WICK_PARTITION_JOBS"] = "1"

    wick = Wick(Ref(maxActiveCumulantRank = maxActiveCumulantRank))
    symbolicAcc: dict[tuple, Fraction] = {}
    maxTerms = 0
    start = perf_counter()

    for offset, ops in enumerate(spinChunk):
        expr = wick.evalSpinString(
            ops,
            connected = connected,
            spinStringIndex = chunkStart + offset,
            spinStringTotal = None,
        )
        maxTerms = max(maxTerms, len(expr))
        accumulateTerms(
            symbolicAcc,
            expr,
        )

    expr = accumulatedExpr(
        symbolicAcc,
        sort = False,
    )
    interner = FactorInterner()
    shardAcc = encodeExprToIntAcc(
        interner,
        expr,
    )

    return {
        "start": chunkStart,
        "count": len(spinChunk),
        "deltas": tuple(interner.deltas),
        "tensors": tuple(interner.tensors),
        "acc": shardAcc,
        "terms": len(expr),
        "max_terms": maxTerms,
        "rss": currentRssKiB(),
        "elapsed": perf_counter() - start,
    }

def _wickPartitionBranchWorker(args):
    """
    Evaluate one top-level Wick partition branch in a worker process.

    Notation:
        P = B_0 \cup P_{\mathrm{rest}}

    Examples:
        A pathological spin string can distribute first-block branches over
        worker processes without enumerating and skipping earlier partitions.
    """
    (
        ops,
        connected,
        maxActiveCumulantRank,
        firstBlocks,
    ) = args
    setProfiling(False)
    wick = Wick(Ref(maxActiveCumulantRank = maxActiveCumulantRank))
    start = perf_counter()
    symbolicAcc: dict[tuple, Fraction] = {}
    for firstBlock in firstBlocks:
        accumulateTerms(
            symbolicAcc,
            wick.evalSpinStringPartitionBranch(
                ops,
                connected,
                firstBlock,
            ),
        )
    expr = accumulatedExpr(
        symbolicAcc,
        sort = False,
    )
    interner = FactorInterner()
    shardAcc = encodeExprToIntAcc(
        interner,
        expr,
    )
    return {
        "branch": firstBlocks[0],
        "count": len(firstBlocks),
        "deltas": tuple(interner.deltas),
        "tensors": tuple(interner.tensors),
        "acc": shardAcc,
        "terms": len(expr),
        "rss": currentRssKiB(),
        "elapsed": perf_counter() - start,
    }

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
    
    if coeff == 1:
        return expr

    if coeff == -1:
        return tuple(
            Term(
                coeff = -term.coeff,
                deltas = term.deltas,
                tensors = term.tensors,
            )
            for term in expr
            if term.coeff != 0
        )

    # Normalise input coefficient to a Fraction.
    c = coeff if isinstance(coeff, Fraction) else Fraction(coeff)
    
    # If the coefficient is zero the scaled expression is zero.
    if c == 0:
        return zero()

    out = []
    for term in expr:
        coeff = c * term.coeff
        if coeff == 0:
            continue
        out.append(
            Term(
                # New coefficient after scaling.
                coeff = coeff,
                # Delta factors symbolic so unchanged.
                deltas = term.deltas,
                # Tensors symbolic so unchanged.
                tensors = term.tensors,
            )
        )

    return tuple(out)

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

    # Accumulate coefficients by symbolic product.
    acc = {}

    for expr in exprs:
        accumulateTerms(acc, expr)
    
    # Convert accumulator back to sorted terms.
    return accumulatedExpr(acc)

def mul(left: Expr, right: Expr) -> Expr:
    """
    Multiply scalar expressions commutatively.

    Notation:
        A B

    Examples:
        mul(delta(i, j), tensor("Gamma1", (u,), (w,)))
        represents \delta^i_j \Gamma^u_w
    """
    with timed("Expr.mul"):
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

def mulRaw(left: Expr, right: Expr) -> Expr:
    """
    Multiply scalar expressions without combining like terms.

    Notation:
        A B

    Examples:
        mulRaw(delta(i, j), tensor("Gamma1", (u,), (w,)))
        returns raw product terms before coefficient collection.
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

    return tuple(out)

def prod(exprs: tuple[Expr, ...]) -> Expr:
    """
    Multiply many scalar expressions.

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
    with timed("Expr.prod"):
        # Begin with unity.
        out = one()
        
        # For each expression multiply it by the accumulated product.
        for expr in exprs:
            out = mulRaw(out, expr)

        acc = {}
        accumulateTerms(acc, out)
        return accumulatedExpr(
            acc,
            sort = False,
        )


def combine(expr: Expr) -> Expr:
    """
    Combine identical scalar products.

    Notation:
        c A + d A = (c + d) A

    Examples:
        (1/2) \Gamma^u_v + (1/2) \Gamma^u_v
        becomes \Gamma^u_v
    """
    with timed("Expr.combine"):
        acc = {}
        accumulateTerms(acc, expr)
        
        return accumulatedExpr(acc)

def accumulateCoeff(
    acc: dict[tuple, Fraction],
    key: tuple,
    coeff: Fraction,
) -> None:
    if coeff == 0:
        return

    old = acc.get(key)

    if old is None:
        acc[key] = coeff
        return

    value = addCoeffValuesFast(
        old,
        coeff,
    )

    if value == 0:
        del acc[key]
    else:
        acc[key] = value 

def accumulateTerms(acc: dict[tuple, Fraction], expr: Expr) -> None:
    """
    Accumulate scalar expression coefficients by symbolic product.

    Notation:
        c A + d A -> acc[A] = c + d
    """
    for term in expr:
        accumulateCoeff(
            acc,
            (term.deltas, term.tensors),
            term.coeff,
        )

def accumulateScaledTerms(acc: dict[tuple, Fraction], expr: Expr, coeff: int | Fraction) -> None:
    """
    Accumulate scaled expression coefficients by symbolic product.

    Notation:
        acc[A] <- acc[A] + c e_A
    """
    if coeff == 1:
        accumulateTerms(acc, expr)
        return

    if coeff == 0:
        return

    c = coeff if isinstance(coeff, Fraction) else Fraction(coeff)

    for term in expr:
        termCoeff = term.coeff

        if termCoeff == 1:
            value = c
        elif termCoeff == -1:
            value = -c
        else:
            value = mulCoeffValuesFast(
                c,
                termCoeff,
            )

        accumulateCoeff(
            acc,
            (term.deltas, term.tensors),
            value,
        )

@cache
def sortedTupleFactors(items: tuple) -> tuple:
    """
    Return a sorted factor tuple with a cache.

    Notation:
        (x_{\pi(1)}, \ldots, x_{\pi(n)}) with x_{\pi(1)} \le \cdots \le x_{\pi(n)}

    Examples:
    """
    return tuple(sorted(items))

def sortedFactors(items) -> tuple:
    """
    Return factors in canonical sorted order.

    Notation:
        \prod_i f_i -> \prod_i f_{\pi(i)}

    Examples:
        Empty and one-factor products are returned without sorting.
    """
    if len(items) < 2:
        return tuple(items)

    return sortedTupleFactors(tuple(items))

def accumulateProductTerms(
    acc: dict[tuple, Fraction],
    factors: tuple[Expr, ...],
    coeff: int | Fraction,
) -> None:
    """
    Accumulate a product of expressions without materialising an intermediate
    expression.

    Notation:
        \text{Accumulator} += c \prod_k E_k

    Examples:
        Used inside Wick partition evaluation to avoid building one Expr per
        Wick partition.
    """
    if coeff == 0:
        return

    c = coeff if isinstance(coeff, Fraction) else Fraction(coeff)

    nonOneFactors = []
    allSingle = True

    for factor in factors:
        if isOneExpr(factor):
            continue

        if len(factor) != 1:
            allSingle = False

        nonOneFactors.append(factor)

    factors = nonOneFactors

    if not factors:
        acc[((), ())] = acc.get(((), ()), ZERO_FRACTION) + c
        return

    if allSingle:
        coeffOut = c
        deltas = []
        tensors = []

        for expr in factors:
            term = expr[0]
            termCoeff = term.coeff

            if termCoeff == 1:
                pass
            elif termCoeff == -1:
                coeffOut = -coeffOut
            else:
                coeffOut = mulCoeffValuesFast(
                    coeffOut,
                    termCoeff,
                )

            if coeffOut == 0:
                return

            deltas.extend(term.deltas)
            tensors.extend(term.tensors)
        
        key = (
            sortedFactors(deltas),
            sortedFactors(tensors),
        )
        accumulateCoeff(
            acc,
            key,
            coeffOut,
        )
        return

    def accumulatePartial(partial: list[tuple[Fraction, tuple[Delta, ...], tuple[Tensor, ...]]]) -> None:
        """
        Accumulate already-expanded product terms.

        Notation:
            acc += partial

        Examples:
            The bounded product expander emits one chunk of completed products.
        """
        for coeffOut, deltas, tensors in partial:
            key = (
                sortedFactors(deltas),
                sortedFactors(tensors),
            )
            accumulateCoeff(
                acc,
                key,
                coeffOut,
            )

    def expandOne(
        partial: list[tuple[Fraction, tuple[Delta, ...], tuple[Tensor, ...]]],
        expr: Expr,
    ) -> list[tuple[Fraction, tuple[Delta, ...], tuple[Tensor, ...]]]:
        """
        Expand one factor over a bounded partial product list.

        Notation:
            P' = P * E

        Examples:
            Product chunks stay below PRODUCT_PARTIAL_TERM_LIMIT terms.
        """
        nextPartial = []

        for leftCoeff, leftDeltas, leftTensors in partial:
            for term in expr:
                termCoeff = term.coeff

                if termCoeff == 1:
                    value = leftCoeff
                elif termCoeff == -1:
                    value = -leftCoeff
                else:
                    value = mulCoeffValuesFast(
                        leftCoeff,
                        termCoeff,
                    )

                if value == 0:
                    continue

                nextPartial.append((
                    value,
                    leftDeltas + term.deltas,
                    leftTensors + term.tensors,
                ))

        return nextPartial

    def process(
        index: int,
        partial: list[tuple[Fraction, tuple[Delta, ...], tuple[Tensor, ...]]],
    ) -> None:
        """
        Expand remaining factors in bounded chunks.

        Notation:
            acc += P * \prod_{k=index} E_k

        Examples:
            A huge active-cumulant product is split into product chunks rather
            than one giant intermediate list or one term-at-a-time recursion.
        """
        if not partial:
            return

        if index == len(factors):
            accumulatePartial(partial)
            return

        expr = factors[index]
        if not expr:
            return

        maxPartial = max(
            1,
            PRODUCT_PARTIAL_TERM_LIMIT // len(expr),
        )

        for start in range(0, len(partial), maxPartial):
            nextPartial = expandOne(
                partial[start:start + maxPartial],
                expr,
            )
            process(
                index + 1,
                nextPartial,
            )

    process(
        0,
        [(
            c,
            (),
            (),
        )],
    )

def compactAccumulator(acc: dict[tuple, Fraction]) -> None:
    """
    Rebuild an accumulator in place to release hash-table slack.

    Notation:
        acc[A] = c -> compact(acc)[A] = c

    Examples:
        evalSpinString calls this when SPINSTRING_ACC_COMPACT_LIMIT is reached.
    """
    if not acc:
        return

    compact = drainedAccumulatedExpr(
        acc,
        sort = False,
    )
    accumulateTerms(
        acc,
        compact,
    )

def maybeCacheSpinString(
    cache: dict,
    key,
    result: Expr,
    limit: int,
) -> None:
    """
    Cache one spin-string expression only if it is small enough.

    Notation:
        |E| <= L -> cache[key] = E

    Examples:
        A small CToV spin string remains cached, but a huge AAToAA spin string
        is returned without being retained by Wick.spinStringCache.
    """
    if limit <= 0 or len(result) <= limit:
        cache[key] = result

def isOneExpr(expr: Expr) -> bool:
    return (
        len(expr) == 1
        and expr[0].coeff == 1
        and not expr[0].deltas
        and not expr[0].tensors
    )

def accumulatedExpr(acc: dict[tuple, Fraction], sort: bool = True) -> Expr:
    """
    Emit a sorted scalar expression from an accumulated coefficient dictionary.

    Notation:
        acc[A] = c -> c A

    Examples:
        Zero coefficients are discarded and remaining terms are sorted.
    """
    if not acc:
        return ()

    if len(acc) == 1:
        ((deltas, tensors), coeff), = acc.items()

        if coeff == 0:
            return ()

        return (Term(
            coeff = coeff,
            deltas = deltas,
            tensors = tensors,
        ),)

    items = sorted(acc.items()) if sort else acc.items()
    out = []

    for (deltas, tensors), coeff in items:
        if coeff == 0:
            continue

        out.append(
            Term(
                coeff = coeff,
                deltas = deltas,
                tensors = tensors,
            )
        )

    return tuple(out)

def drainedAccumulatedExpr(acc: dict[tuple, Fraction], sort: bool = True) -> Expr:
    """
    Emit an expression while emptying the accumulator.

    Notation:
        acc[A] = c -> E = \sum_A c A and acc <- {}

    Examples:
        evalSpinString uses drainedAccumulatedExpr(acc, sort = False) so a
        huge coefficient dictionary is not retained while building the final
        spin-string expression.
    """
    if sort:
        out = accumulatedExpr(
            acc,
            sort = True,
        )
        acc.clear()
        return out

    out = []

    while acc:
        (deltas, tensors), coeff = acc.popitem()

        if coeff == 0:
            continue

        out.append(
            Term(
                coeff = coeff,
                deltas = deltas,
                tensors = tensors,
            )
        )

    return tuple(out)

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
        # Accumulate tensor terms and combine once at the end.
        acc = {}
        
        # Loop over every non-zero permutation coefficient c_\pi of the lower spin indices.
        for lowerPerm, coeff in Spin.coeffs(
            len(upper),
            upperSpins,
            lowerSpins,
        ):
            accumulateTerms(
                acc,
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

        return accumulatedExpr(acc)

    @staticmethod
    @cache
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
        
        # Setup spin-projection linear system by first building Gram matrix.
        gram = spinGram(rank)
        
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

def prewarmSpinCoeffs(maxRank: int = 4) -> None:
    """
    Precompute spin-projection coefficient tables used by active cumulants.

    Notation:
        \lambda^{p_1\sigma_1\cdots p_k\sigma_k}_{q_1\tau_1\cdots q_k\tau_k}
        =
        \sum_{\pi} c_{\pi} \Lambda^{p_1\cdots p_k}_{q_{\pi(1)}\cdots q_{\pi(k)}}.

    Examples:
        R2 generation calls this once per process before hard Wick kernels.
    """
    for rank in range(2, maxRank + 1):
        for upperSpins in cartesianProduct(SPINS, repeat = rank):
            for lowerSpins in cartesianProduct(SPINS, repeat = rank):
                Spin.coeffs(
                    rank,
                    upperSpins,
                    lowerSpins,
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

    def __init__(self, maxActiveCumulantRank: int | None = 4):
        """
        Build reference cumulant rules.

        Notation:
            \Lambda_k = 0 for k > maxActiveCumulantRank

        Examples:
            Ref() keeps active cumulants through \Lambda_4.
            Ref(maxActiveCumulantRank = None) disables active-rank truncation.
        """
        self.maxActiveCumulantRank = maxActiveCumulantRank
        self.activeKappaCache: dict[tuple[tuple[str, Idx, str], ...], Expr] = {}

    def mayHaveNonzeroKappa(self, ops: tuple[Op, ...]) -> bool:
        """
        Test whether a Wick block can have a nonzero cumulant.

        Notation:
            \kappa(B) \ne 0 is possible

        Examples:
            A singleton block is impossible.
            A core create-annihilate pair with equal spin is possible.
            An active rank-5 cumulant is discarded when the rank cap is 4.
        """
        if not ops:
            return False

        spaces = {
            op.idx.space
            for op in ops
        }

        if spaces == {Space.ACTIVE}:
            creators = tuple(
                op
                for op in ops
                if op.kind == "create"
            )
            annihilators = tuple(
                op
                for op in ops
                if op.kind == "annihilate"
            )

            if len(ops) == 2:
                left, right = ops

                return (
                    left.spin == right.spin
                    and left.kind != right.kind
                )

            if len(creators) != len(annihilators):
                return False

            creatorSpins = [op.spin for op in creators]
            annihilatorSpins = [op.spin for op in annihilators]

            if creatorSpins.count(ALPHA) != annihilatorSpins.count(ALPHA):
                return False

            rank = len(creators)

            if rank == 0:
                return False

            if (
                self.maxActiveCumulantRank is not None
                and rank > self.maxActiveCumulantRank
            ):
                return False

            return True

        if len(ops) != 2:
            return False

        left, right = ops

        if left.spin != right.spin:
            return False

        if left.idx.space != right.idx.space:
            return False

        if left.idx.space == Space.CORE:
            return (
                left.kind == "create"
                and right.kind == "annihilate"
            )

        if left.idx.space == Space.VIRTUAL:
            return (
                left.kind == "annihilate"
                and right.kind == "create"
            )

        return False

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

        if not self.mayHaveNonzeroKappa(ops):
            return zero()
        
        # Get the orbital spaces in this cumulant block.
        spaces = {op.idx.space for op in ops}
        
        # Go to active cumulant rules.
        if spaces == {Space.ACTIVE}:
            return self.activeKappaCached(ops)
        
        # If the block is not active only the only non-zero term are two operator 
        # contractions in core or virtual spaces.
        if len(ops) == 2:
            return self.frozenPair(ops[0], ops[1])

        return zero()

    def activeKappaKey(self, ops: tuple[Op, ...]) -> tuple[tuple[str, Idx, str], ...]:
        """
        Return the group-independent cache key for one active cumulant block.

        Notation:
            (a^\dagger_{p\sigma}, a_{q\tau}) -> ((create, p, sigma), (annihilate, q, tau))

        Examples:
            The same active block in different normal-ordered groups shares
            one projected spin-free cumulant.
        """
        return tuple(
            (
                op.kind,
                op.idx,
                op.spin,
            )
            for op in ops
        )

    def activeKappaCached(self, ops: tuple[Op, ...]) -> Expr:
        """
        Evaluate one active cumulant block with memoisation.

        Notation:
            \kappa_A(B)

        Examples:
            Repeated active blocks reuse the first spin-free cumulant projection.
        """
        key = self.activeKappaKey(ops)

        if key not in self.activeKappaCache:
            self.activeKappaCache[key] = self.activeKappa(ops)

        return self.activeKappaCache[key]

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

        if (
            self.maxActiveCumulantRank is not None
            and rank > self.maxActiveCumulantRank
        ):
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
        self.kappaCache: dict[tuple[tuple[str, Idx, str], ...], Expr] = {}
        self.prodCache: dict[tuple[Expr, ...], Expr] = {}
        self.signCache: dict[tuple[tuple[int, ...], ...], int] = {}
        self.blockCache: dict[
            tuple[tuple[str, Idx, str, int], ...],
            tuple[tuple[tuple[int, ...], Expr], ...],
        ] = {}
        self.partitionCache: dict[
            tuple[tuple[str, Idx, str, int], ...],
            tuple[tuple[tuple[tuple[int, ...], Expr], ...], ...],
        ] = {}
        self.spinStringCache: dict[
            tuple[tuple[tuple[str, Idx, str, int], ...], bool],
            Expr,
        ] = {}
        self.shapeBlockCache: dict[
            tuple[tuple[str, Space, str, int], ...],
            tuple[tuple[int, ...], ...],
        ] = {}

        self.partitionPatternCache: dict[
            tuple[tuple[tuple[str, Space, str, int], ...], bool],
            tuple[tuple[int, tuple[tuple[int, ...], ...]], ...],
        ] = {}
        self.progressClassName: str | None = None
        self.progressWorkItem: int | None = None

    def kappaKey(self, ops: tuple[Op, ...]) -> tuple[tuple[str, Idx, str], ...]:
        """
        Return the group-independent cache key for one Wick cumulant block.

        Notation:
            B -> B without normal-order group labels

        Examples:
            The same cumulant block reached from different products reuses the
            first evaluated Ref.kappa value.
        """
        return tuple(
            (
                op.kind,
                op.idx,
                op.spin,
            )
            for op in ops
        )

    def spinStringKey(self, ops: tuple[Op, ...]) -> tuple[tuple[str, Idx, str, int], ...]:
        """
        Return the cache key for one spin string with normalised group labels.

        Notation:
            (g_5, g_5, g_9) -> (g_0, g_0, g_1)

        Examples:
            Spin strings that differ only by group numbering share nonzero
            blocks, viable partitions, and evaluated expressions.
        """
        groups = {}

        return tuple(
            (
                op.kind,
                op.idx,
                op.spin,
                groups.setdefault(op.group, len(groups)),
            )
            for op in ops
        )

    def spinStringShapeKey(
        self,
        ops: tuple[Op, ...],
    ) -> tuple[tuple[str, Space, str, int], ...]:
        """
        Return a structural key for one spin string.

        Notation:
            a^\dagger_{p \sigma} -> (create, space(p), \sigma, group)

        Examples:
            u_l and u_r both become active-space labels in this key.
        """
        groups = {}

        return tuple(
            (
                op.kind,
                op.idx.space,
                op.spin,
                groups.setdefault(op.group, len(groups)),
            )
            for op in ops
        )

    def kappaCached(self, ops: tuple[Op, ...]) -> Expr:
        """
        Evaluate one reference cumulant with memoisation.

        Notation:
            \kappa(B)

        Examples:
            Repeated Wick blocks reuse the first computed cumulant.
        """
        key = self.kappaKey(ops)

        if key not in self.kappaCache:
            with timed("Ref.kappa"):
                self.kappaCache[key] = self.ref.kappa(ops)

        return self.kappaCache[key]

    def prodCached(self, factors: tuple[Expr, ...]) -> Expr:
        """
        Multiply Wick partition factors with memoisation.

        Notation:
            \prod_B \kappa(B)

        Examples:
            Repeated Wick partitions with the same cumulant factors reuse the
            first scalar product expression.
        """
        key = tuple(id(factor) for factor in factors)

        if key not in self.prodCache:
            self.prodCache[key] = prod(factors)

        return self.prodCache[key]

    def eval(self, product: Product, connected: bool = False,) -> Expr:
        """
        Evaluate a product of normal ordered groups and return symbolic terms.

        Notation:
            W(P) = \sum_A c_A A

        Examples:
            Wick(ref).eval(product, connected=True) returns the public symbolic
            expression used by non-hot callers.
        """
        interner, acc = self.evalInt(
            product,
            connected = connected,
        )
        return decodeIntAcc(
            interner,
            acc,
            sort = False,
        )

    def evalInt(self, product: Product, connected: bool = False,) -> tuple[FactorInterner, IntAcc]:
        """
        Evaluate a product of normal ordered groups into an integer accumulator.

        Notation:
            \langle \Phi | \{A\}\{B\}\{C\} \cdots | \Phi \rangle.
            = \sum_{P} sign(P) \prod_{B \in P} \kappa(B).

        Examples:
            product = Product((tau2(i, u, v, a, 0), tau2(x, b, j, w, 1)))
            represents \langle \Phi | \{E^{iu}_{va}\}\{E^{xb}_{jw}\} | \Phi \rangle

            Wick(ref).eval(Product((bra, h, tau)), connected = True)
            evaluates only connected residual contractions.

        The algorithm expands spin-free E operators into spin-orbital
        strings, enumerates Wick partitions, rejects internal contractions,
        applies Ref.kappa to each block, and sums the result.
        """

        # Expand spin-free groups into spin-orbital strings.
        with timed("Wick.expand"):
            spinStrings = self.expand(product)

        spinStringTotal = len(spinStrings)
        spinStart, spinStop, spinPartial = wickEnvRange(
            "WICK_SPIN_START",
            "WICK_SPIN_STOP",
            spinStringTotal,
        )
        selectedSpinStrings = tuple(enumerate(spinStrings[spinStart:spinStop], spinStart + 1))

        jobs = wickEnvPositiveInt("WICK_SPIN_JOBS", 1)
        minSpinStrings = wickEnvPositiveInt("WICK_SPIN_PARALLEL_MIN_SPINSTRINGS", 32)
        minOps = wickEnvPositiveInt("WICK_SPIN_PARALLEL_MIN_OPS", 14)
        chunkSize = wickEnvPositiveInt("WICK_SPIN_CHUNK_SIZE", 1)
        useParallel = (
            jobs > 1
            and (spinPartial or len(selectedSpinStrings) >= minSpinStrings)
            and bool(selectedSpinStrings)
            and len(selectedSpinStrings[0][1]) >= minOps
        )

        if useParallel:
            parentInterner = FactorInterner()
            intAcc: IntAcc = {}
            pool = wickSpinPool(jobs)
            chunks = []

            for first in range(0, len(selectedSpinStrings), chunkSize):
                chunkItems = selectedSpinStrings[first:first + chunkSize]
                chunks.append((
                    chunkItems[0][0],
                    tuple(ops for _index, ops in chunkItems),
                ))

            futures = [
                pool.submit(
                    _wickSpinChunkWorker,
                    (
                        chunkStart,
                        spinChunk,
                        connected,
                        self.ref.maxActiveCumulantRank,
                    ),
                )
                for chunkStart, spinChunk in chunks
            ]
            summaries = []
            doneChunks = 0
            doneSpinStrings = 0
            progressClassName = self.progressClassName
            progressWorkItem = self.progressWorkItem

            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                doneChunks += 1
                doneSpinStrings += summary["count"]

                if progressClassName is not None and progressWorkItem is not None:
                    print(
                        (
                            f"Wick parallel {progressClassName}: Work item: {progressWorkItem}; "
                            f"Spin chunks done: {doneChunks}/{len(chunks)}; "
                            f"Spin strings done: {doneSpinStrings}/{len(selectedSpinStrings)}; "
                            f"RSS: {currentRssKiB()} KiB"
                        ),
                        flush = True,
                    )

            for summary in sorted(summaries, key = lambda item: item["start"]):
                mergeIntShard(
                    parentInterner,
                    intAcc,
                    summary["deltas"],
                    summary["tensors"],
                    summary["acc"],
                )

            if spinPartial:
                raise RuntimeError(
                    (
                        "partial Wick spin-string generation requested: "
                        f"WICK_SPIN_START={spinStart + 1}, WICK_SPIN_STOP={spinStop + 1}"
                    )
                )

            return parentInterner, intAcc

        # Accumulate spin-string terms and combine once.
        parentInterner = FactorInterner()
        intAcc: IntAcc = {}
        for spinStringIndex, ops in selectedSpinStrings:
            self.evalSpinString(
                ops,
                connected = connected,
                spinStringIndex = spinStringIndex,
                spinStringTotal = spinStringTotal,
                targetInterner = parentInterner,
                targetAcc = intAcc,
            )

        if spinPartial:
            raise RuntimeError(
                (
                    "partial Wick spin-string generation requested: "
                    f"WICK_SPIN_START={spinStart + 1}, WICK_SPIN_STOP={spinStop + 1}"
                )
            )

        return parentInterner, intAcc

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

    def nonzeroBlocks(
        self,
        ops: tuple[Op, ...],
    ) -> tuple[tuple[tuple[int, ...], Expr], ...]:
        """
        Return non-internal Wick blocks with nonzero cumulants.

        Notation:
            B such that \kappa(B) \ne 0

        Examples:
            Core and virtual pair contractions are retained.
            Active cumulant blocks above the rank cap are rejected.
        """
        key = self.spinStringKey(ops)

        if key in self.blockCache:
            return self.blockCache[key]

        out = []

        for block in self.candidateBlocks(ops):
            if self.internal(ops, block):
                continue

            blockOps = tuple(
                ops[i]
                for i in block
            )

            value = self.kappaCached(blockOps)

            if not value:
                continue

            out.append((
                block,
                value,
            ))

        self.blockCache[key] = tuple(out)

        return self.blockCache[key]

    def candidateBlocks(self, ops: tuple[Op, ...]) -> tuple[tuple[int, ...], ...]:
        """
        Generate Wick blocks that can have nonzero cumulants.

        Notation:
            B such that \kappa(B) \ne 0 is structurally possible

        Examples:
            Core and virtual candidates are two-operator contractions.
            Active candidates are balanced creator-annihilator blocks up to the
            active cumulant rank cap.
        """
        positions = tuple(range(len(ops)))
        out = []

        out.extend(self.frozenCandidateBlocks(ops, positions))
        out.extend(self.activeCandidateBlocks(ops, positions))

        return tuple(out)

    def frozenCandidateBlocks(
        self,
        ops: tuple[Op, ...],
        positions: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """
        Generate core and virtual two-operator cumulant candidates.

        Notation:
            \kappa(a^\dagger_i, a_j), \kappa(a_a, a^\dagger_b)

        Examples:
            Core create-annihilate pairs and virtual annihilate-create pairs
            with equal spin are retained.
        """
        out = []

        for left, right in combinations(positions, 2):
            leftOp = ops[left]
            rightOp = ops[right]

            if leftOp.spin != rightOp.spin:
                continue

            if leftOp.idx.space != rightOp.idx.space:
                continue

            if leftOp.group == rightOp.group:
                continue

            if (
                leftOp.idx.space == Space.CORE
                and leftOp.kind == "create"
                and rightOp.kind == "annihilate"
            ):
                out.append((left, right))
                continue

            if (
                leftOp.idx.space == Space.VIRTUAL
                and leftOp.kind == "annihilate"
                and rightOp.kind == "create"
            ):
                out.append((left, right))

        return tuple(out)

    def candidateBlocksByShape(
        self,
        ops: tuple[Op, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """
        Generate candidate Wick blocks using only operator shape.

        Notation:

        Examples:
        """
        key = self.spinStringShapeKey(ops)

        if key not in self.shapeBlockCache:
            self.shapeBlockCache[key] = self.candidateBlocks(ops)

        return self.shapeBlockCache[key]

    def activeCandidateBlocks(
        self,
        ops: tuple[Op, ...],
        positions: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """
        Generate active-space cumulant candidates.

        Notation:
            \kappa_A(a^\dagger_{p_1}...a^\dagger_{p_k}a_{q_k}...a_{q_1})

        Examples:
            Active candidates have the same number of creators and
            annihilators and do not exceed the active cumulant rank cap.
        """
        activeCreators = tuple(
            position
            for position in positions
            if (
                ops[position].idx.space == Space.ACTIVE
                and ops[position].kind == "create"
            )
        )
        activeAnnihilators = tuple(
            position
            for position in positions
            if (
                ops[position].idx.space == Space.ACTIVE
                and ops[position].kind == "annihilate"
            )
        )
        maxRank = min(
            len(activeCreators),
            len(activeAnnihilators),
        )

        if self.ref.maxActiveCumulantRank is not None:
            maxRank = min(
                maxRank,
                self.ref.maxActiveCumulantRank,
            )

        if maxRank == 0:
            return ()

        out = []

        for rank in range(1, maxRank + 1):
            for creators in combinations(activeCreators, rank):
                for annihilators in combinations(activeAnnihilators, rank):
                    block = tuple(sorted(creators + annihilators))

                    blockGroups = {
                        ops[position].group
                        for position in block
                    }
                    if len(blockGroups) == 1:
                        continue

                    creatorAlpha = sum(
                        1
                        for position in creators
                        if ops[position].spin == ALPHA
                    )
                    annihilatorAlpha = sum(
                        1
                        for position in annihilators
                        if ops[position].spin == ALPHA
                    )
                    if creatorAlpha != annihilatorAlpha:
                        continue

                    out.append(block)

        return tuple(out)
    
    def viablePartitionPatterns(
        self,
        ops: tuple[Op, ...],
        connected: bool = False,
    ) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
        """
        Generate viable Wick partition patterns and signs.

        This caches by spin/operator shape rather than concrete index names. The
        cached result contains only block positions and fermion signs, not tensor
        expressions.

        Notation:
            shape -> {(sign(P), P)}

        Examples:
            R2 products with the same operator shape but different dummy labels
            reuse the same connected partition patterns.
        """
        with timed("Wick.viablePartitionPatterns"):
            key = (
                self.spinStringShapeKey(ops),
                connected,
            )

            if key in self.partitionPatternCache:
                return self.partitionPatternCache[key]

            groups = tuple(sorted({op.group for op in ops}))
            opGroups = tuple(op.group for op in ops)
            positionBits = tuple(1 << position for position in range(len(ops)))
            groupPairBits = {
                (left, right): 1 << bit
                for bit, (left, right) in enumerate(combinations(groups, 2))
            }

            def blockEdgeMask(block: tuple[int, ...]) -> int:
                """
                Return the group-connectivity edge mask for one candidate block.

                Notation:
                    e(B) = \{(g_i, g_j) : g_i, g_j \in B\}

                Examples:
                """
                blockGroups = tuple(sorted({opGroups[i] for i in block}))

                if len(blockGroups) <= 1:
                    return 0

                out = 0

                for left, right in combinations(blockGroups, 2):
                    out |= groupPairBits[(left, right)]

                return out

            def blockPositionMask(block: tuple[int, ...]) -> int:
                """
                Return the operator-position bit mask for one candidate block.

                Notation:
                    m(B) = \sum_{i \in B} 2^i

                Examples:
                """
                out = 0

                for position in block:
                    out |= positionBits[position]

                return out

            if key[0] not in self.shapeBlockCache:
                self.shapeBlockCache[key[0]] = self.candidateBlocks(ops)

            blocks = tuple(
                (
                    block,
                    blockPositionMask(block),
                    blockEdgeMask(block),
                )
                for block in self.shapeBlockCache[key[0]]
            )

            byPosition: dict[int, list[tuple[tuple[int, ...], int, int]]] = {}

            for block, blockMask, edgeMask in blocks:
                for position in block:
                    byPosition.setdefault(position, []).append((
                        block,
                        blockMask,
                        edgeMask,
                    ))

            fullMask = (1 << len(ops)) - 1
            out: list[tuple[int, tuple[tuple[int, ...], ...]]] = []

            @cache
            def cover(
                remainingMask: int,
                edgeMask: int,
            ) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
                if remainingMask == 0:
                    if connected and not self.edgeMaskConnected(
                        groups,
                        edgeMask,
                    ):
                        return ()

                    return ((
                        1,
                        (),
                    ),)

                first = (remainingMask & -remainingMask).bit_length() - 1
                local = []

                for block, blockMask, blockEdgeMask in byPosition.get(first, ()):
                    if blockMask & remainingMask != blockMask:
                        continue

                    restMask = remainingMask & ~blockMask
                    nextEdgeMask = edgeMask | blockEdgeMask
                    suffixes = cover(
                        restMask,
                        nextEdgeMask,
                    )

                    if not suffixes:
                        continue

                    crossParity = 0

                    for position in block:
                        crossParity ^= (
                            restMask
                            & ((1 << position) - 1)
                        ).bit_count() & 1

                    prefixSign = -1 if crossParity else 1

                    for suffixSign, suffix in suffixes:
                        partition = (block,) + suffix
                        local.append((
                            prefixSign * suffixSign,
                            partition,
                        ))

                return tuple(local)

            out.extend(
                cover(
                    fullMask,
                    0,
                )
            )

            self.partitionPatternCache[key] = tuple(out)

            return self.partitionPatternCache[key]

    def iterViablePartitionPatterns(
        self,
        ops: tuple[Op, ...],
        connected: bool = False,
    ):
        """
        Stream viable Wick partition patterns and signs.

        Notation:
            yield (sign(P), P) without materialising all P.

        Examples:
            evalSpinString uses this for large R2 spin strings so one work item
            does not allocate a giant partition tuple before accumulation.
        """
        with timed("Wick.iterViablePartitionPatterns"):
            groups = tuple(sorted({op.group for op in ops}))
            opGroups = tuple(op.group for op in ops)
            positionBits = tuple(1 << position for position in range(len(ops)))
            groupPairBits = {
                (left, right): 1 << bit
                for bit, (left, right) in enumerate(combinations(groups, 2))
            }

            def blockEdgeMask(block: tuple[int, ...]) -> int:
                blockGroups = tuple(sorted({opGroups[i] for i in block}))

                if len(blockGroups) <= 1:
                    return 0

                out = 0

                for left, right in combinations(blockGroups, 2):
                    out |= groupPairBits[(left, right)]

                return out

            def blockPositionMask(block: tuple[int, ...]) -> int:
                out = 0

                for position in block:
                    out |= positionBits[position]

                return out

            shapeKey = self.spinStringShapeKey(ops)
            if shapeKey not in self.shapeBlockCache:
                self.shapeBlockCache[shapeKey] = self.candidateBlocks(ops)

            blocks = tuple(
                (
                    block,
                    blockPositionMask(block),
                    blockEdgeMask(block),
                )
                for block in self.shapeBlockCache[shapeKey]
            )

            byPosition: dict[int, list[tuple[tuple[int, ...], int, int]]] = {}

            for block, blockMask, edgeMask in blocks:
                for position in block:
                    byPosition.setdefault(position, []).append((
                        block,
                        blockMask,
                        edgeMask,
                    ))

            fullMask = (1 << len(ops)) - 1

            def iterCover(
                remainingMask: int,
                edgeMask: int,
            ):
                if remainingMask == 0:
                    if connected and not self.edgeMaskConnected(
                        groups,
                        edgeMask,
                    ):
                        return

                    yield (
                        1,
                        (),
                    )
                    return

                first = (remainingMask & -remainingMask).bit_length() - 1

                for block, blockMask, blockEdgeMask in byPosition.get(first, ()):
                    if blockMask & remainingMask != blockMask:
                        continue

                    restMask = remainingMask & ~blockMask
                    crossParity = 0

                    for position in block:
                        crossParity ^= (
                            restMask
                            & ((1 << position) - 1)
                        ).bit_count() & 1

                    prefixSign = -1 if crossParity else 1
                    nextEdgeMask = edgeMask | blockEdgeMask

                    if (
                        SPINSTRING_STREAM_SUFFIX_CACHE_OP_LIMIT > 0
                        and restMask.bit_count() <= SPINSTRING_STREAM_SUFFIX_CACHE_OP_LIMIT
                    ):
                        suffixes = coverCached(
                            restMask,
                            nextEdgeMask,
                        )
                    else:
                        suffixes = iterCover(
                            restMask,
                            nextEdgeMask,
                        )

                    for suffixSign, suffix in suffixes:
                        yield (
                            prefixSign * suffixSign,
                            (block,) + suffix,
                        )

            @cache
            def coverCached(
                remainingMask: int,
                edgeMask: int,
            ) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
                return tuple(iterCover(remainingMask, edgeMask))

            yield from iterCover(
                fullMask,
                0,
            )

    def iterNonzeroPartitionPatterns(
        self,
        ops: tuple[Op, ...],
        blockValues: dict[tuple[int, ...], Expr],
        connected: bool = False,
    ):
        """
        Stream concrete nonzero Wick partition patterns and signs.

        Notation:
            yield (sign(P), P) with \kappa(B) != 0 for every B in P.

        Examples:
            Large active-space R2 spin strings use this to avoid materialising
            partitions containing blocks that vanish for the concrete indices.
        """
        with timed("Wick.iterNonzeroPartitionPatterns"):
            groups = tuple(sorted({op.group for op in ops}))
            opGroups = tuple(op.group for op in ops)
            positionBits = tuple(1 << position for position in range(len(ops)))
            groupPairBits = {
                (left, right): 1 << bit
                for bit, (left, right) in enumerate(combinations(groups, 2))
            }

            def blockEdgeMask(block: tuple[int, ...]) -> int:
                blockGroups = tuple(sorted({opGroups[i] for i in block}))

                if len(blockGroups) <= 1:
                    return 0

                out = 0

                for left, right in combinations(blockGroups, 2):
                    out |= groupPairBits[(left, right)]

                return out

            def blockPositionMask(block: tuple[int, ...]) -> int:
                out = 0

                for position in block:
                    out |= positionBits[position]

                return out

            rawBlocks = self.candidateBlocksByShape(ops)
            blocks = []

            for block in rawBlocks:
                blockOps = tuple(
                    ops[i]
                    for i in block
                )
                value = self.kappaCached(blockOps)

                if not value:
                    continue

                blockValues[block] = value
                blocks.append((
                    block,
                    blockPositionMask(block),
                    blockEdgeMask(block),
                ))

            byPosition: dict[int, list[tuple[tuple[int, ...], int, int]]] = {}

            for block, blockMask, edgeMask in blocks:
                for position in block:
                    byPosition.setdefault(position, []).append((
                        block,
                        blockMask,
                        edgeMask,
                    ))

            fullMask = (1 << len(ops)) - 1

            def cover(
                remainingMask: int,
                edgeMask: int,
            ):
                if remainingMask == 0:
                    if connected and not self.edgeMaskConnected(
                        groups,
                        edgeMask,
                    ):
                        return

                    yield (
                        1,
                        (),
                    )
                    return

                first = (remainingMask & -remainingMask).bit_length() - 1

                for block, blockMask, blockEdgeMask in byPosition.get(first, ()):
                    if blockMask & remainingMask != blockMask:
                        continue

                    restMask = remainingMask & ~blockMask
                    suffixes = cover(
                        restMask,
                        edgeMask | blockEdgeMask,
                    )

                    crossParity = 0

                    for position in block:
                        crossParity ^= (
                            restMask
                            & ((1 << position) - 1)
                        ).bit_count() & 1

                    prefixSign = -1 if crossParity else 1

                    for suffixSign, suffix in suffixes:
                        yield (
                            prefixSign * suffixSign,
                            (block,) + suffix,
                        )

            yield from cover(
                fullMask,
                0,
            )

    def nonzeroPartitionBranches(
        self,
        ops: tuple[Op, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """
        Yield top-level nonzero Wick partition branches.

        Notation:
            P = B_0 \cup P_{\mathrm{rest}}

        Examples:
            A pathological spin string distributes first-block branches over
            workers instead of making one worker enumerate all partitions.
        """
        if not ops:
            return ()

        out = []
        for block in self.candidateBlocksByShape(ops):
            if 0 not in block:
                continue

            blockOps = tuple(ops[i] for i in block)
            if self.kappaCached(blockOps):
                out.append(block)

        return tuple(out)

    def evalSpinStringPartitionBranch(
        self,
        ops: tuple[Op, ...],
        connected: bool,
        firstBlock: tuple[int, ...],
    ) -> Expr:
        """
        Evaluate one top-level Wick partition branch.

        Notation:
            \sum_{P: B_0 \in P} \sign(P) \prod_{B \in P} \kappa(B)

        Examples:
            WICK_PARTITION_JOBS uses this for one branch of a large spin string.
        """
        groups = tuple(sorted({op.group for op in ops}))
        opGroups = tuple(op.group for op in ops)
        positionBits = tuple(1 << position for position in range(len(ops)))
        groupPairBits = {
            (left, right): 1 << bit
            for bit, (left, right) in enumerate(combinations(groups, 2))
        }

        def blockEdgeMask(block: tuple[int, ...]) -> int:
            blockGroups = tuple(sorted({opGroups[i] for i in block}))
            if len(blockGroups) <= 1:
                return 0

            out = 0
            for left, right in combinations(blockGroups, 2):
                out |= groupPairBits[(left, right)]
            return out

        def blockPositionMask(block: tuple[int, ...]) -> int:
            out = 0
            for position in block:
                out |= positionBits[position]
            return out

        blockValues: dict[tuple[int, ...], Expr] = {}
        blocks = []

        for block in self.candidateBlocksByShape(ops):
            blockOps = tuple(ops[i] for i in block)
            value = self.kappaCached(blockOps)
            if not value:
                continue

            blockValues[block] = value
            blocks.append((
                block,
                blockPositionMask(block),
                blockEdgeMask(block),
            ))

        byPosition: dict[int, list[tuple[tuple[int, ...], int, int]]] = {}
        blockMeta = {}

        for block, blockMask, edgeMask in blocks:
            blockMeta[block] = (
                blockMask,
                edgeMask,
            )
            for position in block:
                byPosition.setdefault(position, []).append((
                    block,
                    blockMask,
                    edgeMask,
                ))

        firstMeta = blockMeta.get(firstBlock)
        firstValue = blockValues.get(firstBlock)
        if firstMeta is None or not firstValue:
            return ()

        firstMask, firstEdgeMask = firstMeta
        fullMask = (1 << len(ops)) - 1
        restAfterFirst = fullMask & ~firstMask
        crossParity = 0
        for position in firstBlock:
            crossParity ^= (
                restAfterFirst
                & ((1 << position) - 1)
            ).bit_count() & 1
        firstSign = -1 if crossParity else 1

        def cover(
            remainingMask: int,
            edgeMask: int,
        ):
            if remainingMask == 0:
                if connected and not self.edgeMaskConnected(
                    groups,
                    edgeMask,
                ):
                    return

                yield (
                    1,
                    (),
                )
                return

            first = (remainingMask & -remainingMask).bit_length() - 1

            for block, blockMask, blockEdgeMask in byPosition.get(first, ()):
                if blockMask & remainingMask != blockMask:
                    continue

                restMask = remainingMask & ~blockMask
                suffixes = cover(
                    restMask,
                    edgeMask | blockEdgeMask,
                )

                crossParity = 0
                for position in block:
                    crossParity ^= (
                        restMask
                        & ((1 << position) - 1)
                    ).bit_count() & 1

                prefixSign = -1 if crossParity else 1

                for suffixSign, suffix in suffixes:
                    yield (
                        prefixSign * suffixSign,
                        (block,) + suffix,
                    )

        acc = {}
        nextAccCompact = SPINSTRING_ACC_COMPACT_LIMIT
        for suffixSign, suffix in cover(
            restAfterFirst,
            firstEdgeMask,
        ):
            factors = [firstValue]
            ok = True

            for block in suffix:
                value = blockValues.get(block)
                if not value:
                    ok = False
                    break
                factors.append(value)

            if not ok:
                continue

            accumulateProductTerms(
                acc,
                tuple(factors),
                firstSign * suffixSign,
            )

            if nextAccCompact > 0 and len(acc) >= nextAccCompact:
                compactAccumulator(acc)
                nextAccCompact = len(acc) + SPINSTRING_ACC_COMPACT_LIMIT

        return drainedAccumulatedExpr(
            acc,
            sort = False,
        )

    def evalSpinString(
        self,
        ops: tuple[Op, ...],
        connected: bool = False,
        spinStringIndex: int | None = None,
        spinStringTotal: int | None = None,
        targetInterner: FactorInterner | None = None,
        targetAcc: IntAcc | None = None,
    ) -> Expr:
        """
        Evaluate one spin-orbital string by generalised Wick's theorem.

        A partition is rejected if any block is internal to one normal-ordered
        group due to normal ordering or if Ref.kappa(block) is zero, or, when 
        connected is requested, if the partition does not connect every GNO group.

        Notation:
            \sum_P \sign(P) \prod_{B \in P} \kappa(B)

        Examples:
            Input string:
            \{a^\dagger_{i\alpha}a_{u\alpha}\}
            \{a^\dagger_{x\alpha}a_{a\alpha}\}

            The block \kappa(a^\dagger_{i\alpha}, a_{u\alpha}) is
            rejected because it is internal to one group. The block
            \kappa(a_{u\alpha}, a^\dagger_{x\alpha}) is allowed if it
            connects different groups. 

        In the second order residual, \{T^2\} = \{TT\}, both T's are given the 
        same group id, thus we may have internal blocks to \{T^2\} which are
        rejected by the same rule.
        """
        with timed("Wick.evalSpinString"):
            progressClassName = self.progressClassName
            progressWorkItem = self.progressWorkItem
            showProgress = progressClassName is not None and progressWorkItem is not None

            def printProgress(
                phase: str,
                partitionIndex: int | None = None,
                partitionTotal: int | None = None,
                accSize: int | None = None,
                terms: int | None = None,
            ) -> None:
                """
                Print one nested Wick progress line.

                Notation:
                    R2 work item -> spin string -> Wick partitions.

                Examples:
                    evalSpinString prints every WICK_PROGRESS_PARTITION_INTERVAL partitions.
                """
                if not showProgress:
                    return

                spin = "?"
                if spinStringIndex is not None and spinStringTotal is not None:
                    spin = f"{spinStringIndex}/{spinStringTotal}"

                parts = [
                    f"  Wick progress {progressClassName}:",
                    f"Work item: {progressWorkItem};",
                    f"Spin string index: {spin};",
                    f"Phase: {phase};",
                ]
                if partitionIndex is not None:
                    if partitionTotal is None:
                        parts.append(f"Partition index: {partitionIndex};")
                    else:
                        parts.append(f"Partition index: {partitionIndex}/{partitionTotal};")
                if accSize is not None:
                    parts.append(f"Accumulator size: {accSize};")
                if terms is not None:
                    parts.append(f"Terms: {terms};")
                parts.append(f"RSS: {currentRssKiB()} KiB")
                print(
                    " ".join(parts),
                    flush = True,
                )

            printProgress("spin_string_start")
            key = (self.spinStringKey(ops), connected)
            if key in self.spinStringCache:
                result = self.spinStringCache[key]
                printProgress(
                    "spin_string_done",
                    terms = len(result),
                )
                return result

            if not self.maybeSpinBalanced(ops):
                self.spinStringCache[key] = ()
                printProgress(
                    "spin_string_done",
                    terms = 0,
                )
                return ()

            if connected and not self.maybeConnected(ops):
                self.spinStringCache[key] = ()
                printProgress(
                    "spin_string_done",
                    terms = 0,
                )
                return ()

            partitionJobs = wickEnvPositiveInt("WICK_PARTITION_JOBS", 1)
            partitionMinPartitions = wickEnvPositiveInt(
                "WICK_PARTITION_PARALLEL_MIN_PARTITIONS",
                100_000,
            )
            if (
                partitionJobs > 1
            ):
                branches = self.nonzeroPartitionBranches(ops)

                if branches and len(branches) <= partitionMinPartitions:
                    pool = wickSpinPool(partitionJobs)
                    branchChunkSize = wickEnvPositiveInt(
                        "WICK_PARTITION_BRANCH_CHUNK_SIZE",
                        1,
                    )
                    branchChunks = tuple(
                        branches[first:first + branchChunkSize]
                        for first in range(0, len(branches), branchChunkSize)
                    )
                    futures = [
                        pool.submit(
                            _wickPartitionBranchWorker,
                            (
                                ops,
                                connected,
                                self.ref.maxActiveCumulantRank,
                                branchChunk,
                            ),
                        )
                        for branchChunk in branchChunks
                    ]
                    interner = FactorInterner()
                    intAcc: IntAcc = {}
                    doneBranches = 0

                    for future in as_completed(futures):
                        summary = future.result()
                        mergeIntShard(
                            interner,
                            intAcc,
                            summary["deltas"],
                            summary["tensors"],
                            summary["acc"],
                        )
                        doneBranches += 1

                        printProgress(
                            "partition_branches",
                            partitionIndex = doneBranches,
                            partitionTotal = len(branchChunks),
                            terms = summary["terms"],
                        )

                    if targetInterner is not None and targetAcc is not None:
                        mergeIntShard(
                            targetInterner,
                            targetAcc,
                            tuple(interner.deltas),
                            tuple(interner.tensors),
                            intAcc,
                        )
                        printProgress(
                            "spin_string_done",
                            partitionIndex = len(branchChunks),
                            partitionTotal = len(branchChunks),
                            terms = len(intAcc),
                        )
                        return ()
                    result = decodeIntAcc(
                        interner,
                        intAcc,
                        sort = False,
                    )
                    maybeCacheSpinString(
                        self.spinStringCache,
                        key,
                        result,
                        SPINSTRING_CACHE_TERM_LIMIT,
                    )
                    printProgress(
                        "spin_string_done",
                        partitionIndex = len(branchChunks),
                        partitionTotal = len(branchChunks),
                        terms = len(result),
                    )
                    return result

            acc = {}
            blockValues = {}
            nextAccCompact = SPINSTRING_ACC_COMPACT_LIMIT
            if (
                SPINSTRING_STREAM_OP_LIMIT > 0
                and len(ops) < SPINSTRING_STREAM_OP_LIMIT
            ):
                patterns = self.viablePartitionPatterns(
                    ops,
                    connected = connected,
                )
            else:
                patterns = self.iterNonzeroPartitionPatterns(
                    ops,
                    blockValues,
                    connected = connected,
                )
            partitionTotal = len(patterns) if hasattr(patterns, "__len__") else None

            for partitionIndex, (sign, blocks) in enumerate(patterns, 1):
                factors = []
                ok = True

                for block in blocks:
                    value = blockValues.get(block)
                    if value is None:
                        blockOps = tuple(
                            ops[i]
                            for i in block
                        )
                        value = self.kappaCached(blockOps)
                        blockValues[block] = value

                    if not value:
                        ok = False
                        break

                    factors.append(value)

                if not ok:
                    continue

                accumulateProductTerms(
                    acc,
                    tuple(factors),
                    sign,
                )
                if nextAccCompact > 0 and len(acc) >= nextAccCompact:
                    compactAccumulator(acc)
                    nextAccCompact = len(acc) + SPINSTRING_ACC_COMPACT_LIMIT
                if partitionIndex % WICK_PROGRESS_PARTITION_INTERVAL == 0:
                    printProgress(
                        "partitions",
                        partitionIndex = partitionIndex,
                        partitionTotal = partitionTotal,
                        accSize = len(acc),
                    )

            result = drainedAccumulatedExpr(
                acc,
                sort = False,
            )
            if targetInterner is not None and targetAcc is not None:
                shardInterner = FactorInterner()
                shardAcc = encodeExprToIntAcc(
                    shardInterner,
                    result,
                )
                mergeIntShard(
                    targetInterner,
                    targetAcc,
                    tuple(shardInterner.deltas),
                    tuple(shardInterner.tensors),
                    shardAcc,
                )
                maybeCacheSpinString(
                    self.spinStringCache,
                    key,
                    result,
                    SPINSTRING_CACHE_TERM_LIMIT,
                )
                printProgress(
                    "spin_string_done",
                    partitionIndex = partitionIndex if "partitionIndex" in locals() else 0,
                    partitionTotal = partitionTotal,
                    terms = len(result),
                )
                return ()
            maybeCacheSpinString(
                self.spinStringCache,
                key,
                result,
                SPINSTRING_CACHE_TERM_LIMIT,
            )
            printProgress(
                "spin_string_done",
                partitionIndex = partitionIndex if "partitionIndex" in locals() else 0,
                partitionTotal = partitionTotal,
                terms = len(result),
            )
            return result

    def maybeSpinBalanced(
        self,
        ops: tuple[Op, ...],
    ) -> bool:
        """
        Cheaply test whether a spin string can possibly fully contract.

        Notation:
            n_c(X, \sigma) = n_a(X, \sigma),
            where c is creation and a is annihilation.

        Examples:
            If active alpha creators outnumber active alpha annihilators, every
            active cumulant partition is zero.
        """
        counts: dict[tuple[Space, str], int] = {}

        for op in ops:
            key = (
                op.idx.space,
                op.spin,
            )
            delta = 1 if op.kind == "create" else -1
            counts[key] = counts.get(key, 0) + delta

        return all(value == 0 for value in counts.values())

    def canContract(self, left: Op, right: Op) -> bool:
        """
        Cheaply test whether two spin operators could share a Wick block.

        Notation:

        Examples:
            Different spins and orbital spaces cannot form a frozen pair.
        """
        if left.group == right.group:
            return False

        if left.idx.space != right.idx.space:
            return False

        if left.idx.space == Space.ACTIVE:
            return left.kind != right.kind

        if left.spin != right.spin:
            return False

        if left.idx.space == Space.CORE:
            return (
                left.kind == "create"
                and right.kind == "annihilate"
            ) or (
                left.kind == "annihilate"
                and right.kind == "create"
            )

        if left.idx.space == Space.VIRTUAL:
            return (
                left.kind == "annihilate"
                and right.kind == "create"
            ) or (
                left.kind == "create"
                and right.kind == "annihilate"
            )

        return False

    def maybeConnected(
        self,
        ops: tuple[Op, ...],
    ) -> bool:
        """
        Cheaply test whether a spin string can possibly have a connected Wick
        partition.

        This is only a prefilter. A true result does not imply that a valid
        connected partition exists.

        Notation:

        Examples:
        """
        groups = tuple(sorted({op.group for op in ops}))
        if len(groups) <= 1:
            return True

        parent = {group: group for group in groups}

        def find(group: int) -> int:
            while parent[group] != group:
                parent[group] = parent[parent[group]]
                group = parent[group]
            return group

        def union(left: int, right: int) -> None:
            leftRoot = find(left)
            rightRoot = find(right)
            if leftRoot != rightRoot:
                parent[rightRoot] = leftRoot

        for i, left in enumerate(ops):
            for right in ops[i + 1:]:
                if self.canContract(left, right):
                    union(left.group, right.group)

        root = find(groups[0])
        return all(find(group) == root for group in groups)

    def blockEdgeMask(
        self,
        ops: tuple[Op, ...],
        block: tuple[int, ...],
        groups: tuple[int, ...],
    ) -> int:
        """
        Return the group-connectivity edge mask for one Wick block.

        Notation:

        Examples:
        """
        blockGroups = tuple(sorted({ops[i].group for i in block}))

        if len(blockGroups) <= 1:
            return 0

        groupPairs = []
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                groupPairs.append((
                    left,
                    right,
                ))

        out = 0

        for i, pair in enumerate(groupPairs):
            if pair[0] in blockGroups and pair[1] in blockGroups:
                out |= 1 << i

        return out

    def edgeMaskConnected(
        self,
        groups: tuple[int, ...],
        edgeMask: int,
    ) -> bool:
        """
        Test whether a group-edge mask connects all normal-ordered groups.

        Notation:

        Examples:
            For groups (0, 1, 2), edges 0-1 and 1-2 are connected.
        """
        if len(groups) <= 1:
            return True

        parent = {
            group: group
            for group in groups
        }

        def find(group: int) -> int:
            while parent[group] != group:
                parent[group] = parent[parent[group]]
                group = parent[group]

            return group

        def union(left: int, right: int) -> None:
            leftRoot = find(left)
            rightRoot = find(right)

            if leftRoot != rightRoot:
                parent[rightRoot] = leftRoot

        bit = 0

        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                if edgeMask & (1 << bit):
                    union(
                        left,
                        right,
                    )

                bit += 1

        root = find(groups[0])

        return all(
            find(group) == root
            for group in groups
        )

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
        if partition not in self.signCache:
            self.signCache[partition] = partitionSign(partition)

        return self.signCache[partition]

    def connected(self, ops: tuple[Op, ...], blocks: tuple[tuple[int, ...], ...]) -> bool:
        """ 
        Test whether a Wick partition connects all normal-ordered groups. A partition is 
        connected when the graph whose vertices are GNO group ids and whose edges are Wick 
        blocks has one connected component.

        Notation:

        Examples:
        """
        groups = tuple(sorted({op.group for op in ops}))
        if len(groups) <= 1:
            return True

        parent = {group: group for group in groups}

        def find(group: int) -> int:
            while parent[group] != group:
                parent[group] = parent[parent[group]]
                group = parent[group]
            return group

        def union(left: int, right: int) -> None:
            leftRoot = find(left)
            rightRoot = find(right)
            if leftRoot != rightRoot:
                parent[rightRoot] = leftRoot

        for block in blocks:
            blockGroups = tuple(sorted({ops[i].group for i in block}))
            if len(blockGroups) <= 1:
                continue
            first = blockGroups[0]
            for group in blockGroups[1:]:
                union(first, group)

        root = find(groups[0])
        return all(find(group) == root for group in groups)

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
