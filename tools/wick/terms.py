from __future__ import annotations

import argparse
import json
from fractions import Fraction
import os
from typing import Any
import sys
from time import perf_counter

from core import Idx, Space, Term
from equations import configureWick, outputExpr, residualExpr
from specs import EXCITATIONS, ExcitationSpec, OverlapBlockSpec, availableBlocks, availableExcitations, overlapBlock

SPACE_KIND = {
    Space.CORE: 0,
    Space.ACTIVE: 1,
    Space.VIRTUAL: 2,
}

SPACE_NAMES = {
    "core": 0,
    "active": 1,
    "virtual": 2,
}

TENSOR_KIND = {
    "Gamma1": 0,
    "Theta": 1,
    "f": 2,
    "g": 3,
    "Lambda2": 4,
    "Lambda3": 5,
    "Lambda4": 6,
    "t1": 8,
    "t2": 9,
}

PROFILE = os.environ.get("WICK_PROFILE", "") not in ("", "0", "false", "False")

def indexKey(idx: Idx) -> tuple[str, Space]:
    """
    Return the local key for one symbolic orbital index.

    Notation:
        p in C/A/V -> (name, space)

    Examples:
        A("u") and C("u") would be different keys, and are rejected later
        because the compact JSON representation addresses indices by name.
    """
    return (
        idx.name,
        idx.space,
    )

def encodeSpace(space: Space) -> int:
    """
    Encode one orbital space.

    Notation:
        C -> 0, A -> 1, V -> 2

    Examples:
        Space.ACTIVE becomes 1.
    """
    if space not in SPACE_KIND:
        raise ValueError(f"unknown space {space}")

    return SPACE_KIND[space]

def freeIndices(spec: ExcitationSpec) -> tuple[Idx, ...]:
    """
    Return residual free indices in unpacking order.

    Notation:
        R^{creators}_{annihilators}

    Examples:
        CToA gives (u, i).
    """
    return spec.creators + spec.annihilators

def renamedFreeIndices(
    spec: ExcitationSpec,
    names: tuple[str, ...],
) -> tuple[Idx, ...]:
    """
    Return one excitation's free indices with overlap-block names.

    Notation: class spaces + block-local names -> block-local free indices
    Examples: AAToAV with names ("x", "b", "v", "w") gives x in A, b in V, v in A, w in A.
    """
    base = spec.creators + spec.annihilators

    if len(base) != len(names):
        raise ValueError(f"wrong number of free-index names for {spec.name}")

    return tuple(Idx(name, idx.space) for name, idx in zip(names, base))

def overlapFreeIndices(block: OverlapBlockSpec) -> tuple[Idx, ...]:
    """
    Return overlap free indices in left-then-right unpacking order.

    Notation: <tau_L | tau_R> -> L creators, L annihilators, R creators, R annihilators
    Examples: C1 gives u, i, v, j.
    """
    left = EXCITATIONS[block.left]
    right = EXCITATIONS[block.right]

    return renamedFreeIndices(left, block.unpack[0]) + renamedFreeIndices(right, block.unpack[1])

def termIndices(term: Term) -> tuple[Idx, ...]:
    """
    Return every index occurrence in one symbolic term.

    Notation:
        c delta tensor tensor -> all indices appearing in those factors

    Examples:
        A Lambda2 factor contributes its upper and lower indices.
    """
    out = []

    for delta in term.deltas:
        out.append(delta.left)
        out.append(delta.right)

    for tensor in term.tensors:
        out.extend(tensor.upper)
        out.extend(tensor.lower)

    return tuple(out)

def addIndex(
    indices: list[Idx],
    indexIds: dict[tuple[str, Space], int],
    spacesByName: dict[str, Space],
    idx: Idx,
) -> None:
    """
    Add one index to the class-local index table.

    Notation:
        index id = position in class-local index list

    Examples:
        The first free index gets id 0.
    """
    oldSpace = spacesByName.get(idx.name)

    if oldSpace is not None and oldSpace != idx.space:
        raise ValueError(
            f"index name {idx.name!r} occurs in both {oldSpace} and {idx.space}"
        )

    spacesByName[idx.name] = idx.space
    key = indexKey(idx)

    if key in indexIds:
        return

    indexIds[key] = len(indices)
    indices.append(idx)

def classIndexTable(
    spec: ExcitationSpec,
    expr: tuple[Term, ...],
) -> tuple[tuple[Idx, ...], dict[tuple[str, Space], int]]:
    """
    Build one residual-class index table.

    Free indices are placed first. Dummy indices are then added in first
    occurrence order over the generated symbolic terms.

    Notation:
        indices = free + dummies

    Examples:
        CToA starts with u, i before any Hamiltonian or amplitude dummies.
    """
    indices: list[Idx] = []
    indexIds: dict[tuple[str, Space], int] = {}
    spacesByName: dict[str, Space] = {}

    for idx in freeIndices(spec):
        addIndex(
            indices,
            indexIds,
            spacesByName,
            idx,
        )

    for term in expr:
        for idx in termIndices(term):
            addIndex(
                indices,
                indexIds,
                spacesByName,
                idx,
            )

    return (
        tuple(indices),
        indexIds,
    )

def overlapIndexTable(
    block: OverlapBlockSpec,
    expr: tuple[Term, ...],
) -> tuple[tuple[Idx, ...], dict[tuple[str, Space], int]]:
    """
    Build one overlap-block index table.

    Free indices are placed first. Dummy indices are then added in first occurrence order over the
    generated symbolic terms.

    Notation: indices = left free + right free + dummies
    Examples: C1 starts with u, i, v, j before any dummy indices.
    """
    indices: list[Idx] = []
    indexIds: dict[tuple[str, Space], int] = {}
    spacesByName: dict[str, Space] = {}

    for idx in overlapFreeIndices(block):
        addIndex(
            indices,
            indexIds,
            spacesByName,
            idx,
        )

    for term in expr:
        for idx in termIndices(term):
            addIndex(
                indices,
                indexIds,
                spacesByName,
                idx,
            )

    return (
        tuple(indices),
        indexIds,
    )

def indexId(
    idx: Idx,
    indexIds: dict[tuple[str, Space], int],
) -> int:
    """
    Return the encoded integer id for one symbolic index.

    Notation:
        u -> 0, i -> 1, ...

    Examples:
        Tensor indices are encoded by class-local ids.
    """
    return indexIds[indexKey(idx)]

def encodeCoeff(coeff) -> list[int]:
    """
    Encode one rational coefficient.

    Notation:
        a / b -> [a, b]

    Examples:
        Fraction(-1, 2) becomes [-1, 2].
    """
    value = Fraction(coeff)

    return [
        int(value.numerator),
        int(value.denominator),
    ]

def encodeLoops(
    term: Term,
    indexIds: dict[tuple[str, Space], int],
    freeIds: set[int],
) -> list[int]:
    """
    Encode the dummy-loop indices for one term.

    Notation:
        loops = term indices minus residual free indices

    Examples:
        A dummy active index x becomes its class-local integer id.
    """
    seen = set()
    loops = []

    for idx in termIndices(term):
        encoded = indexId(
            idx,
            indexIds,
        )

        if encoded in freeIds or encoded in seen:
            continue

        seen.add(encoded)
        loops.append(encoded)

    return loops

def encodeDelta(
    delta,
    indexIds: dict[tuple[str, Space], int],
) -> list[int]:
    """
    Encode one Kronecker delta.

    Notation:
        delta^p_q -> [p, q]

    Examples:
        delta(i, j) becomes [id(i), id(j)].
    """
    return [
        indexId(delta.left, indexIds),
        indexId(delta.right, indexIds),
    ]

def encodeTensor(
    tensor,
    indexIds: dict[tuple[str, Space], int],
) -> list[Any]:
    """
    Encode one tensor factor.

    Notation:
        tensor -> [kind, upper_ids, lower_ids]

    Examples:
        Lambda2^{ux}_{vw} becomes [4, [u, x], [v, w]].
        t1^q_p becomes [8, [q], [p]].
    """
    if tensor.name not in TENSOR_KIND:
        raise ValueError(f"unknown tensor {tensor.name}")

    return [
        TENSOR_KIND[tensor.name],
        [
            indexId(
                idx,
                indexIds,
            )
            for idx in tensor.upper
        ],
        [
            indexId(
                idx,
                indexIds,
            )
            for idx in tensor.lower
        ],
    ]

def encodeTerm(
    term: Term,
    indexIds: dict[tuple[str, Space], int],
    freeIds: set[int],
) -> list[Any]:
    """
    Encode one symbolic term.

    Layout:
        [coeff, loops, deltas, tensors]

    Examples:
        [[1, 2], [2, 3], [], [[4, [0, 2], [3, 1]], [8, [2], [0]]]]
    """
    return [
        encodeCoeff(term.coeff),
        encodeLoops(
            term,
            indexIds,
            freeIds,
        ),
        [
            encodeDelta(
                delta,
                indexIds,
            )
            for delta in term.deltas
        ],
        [
            encodeTensor(
                tensor,
                indexIds,
            )
            for tensor in term.tensors
        ],
    ]

def residualClassTerms(name: str, order: int) -> dict[str, Any]:
    """
    Encode one residual class as compact JSON data.

    Notation:
        R_mu^(order) for one ExcitationClass

    Examples:
        residualClassTerms("CToA", 1) emits the first-order CToA term table.
    """
    start = perf_counter()
    spec = EXCITATIONS[name]
    expr = tuple(
        residualExpr(
            name,
            order = order,
        )
    )

    indices, indexIds = classIndexTable(
        spec,
        expr,
    )

    free = [
        indexId(
            idx,
            indexIds,
        )
        for idx in freeIndices(spec)
    ]

    freeIds = set(free)

    out = {
        "indices": [
            [
                idx.name,
                encodeSpace(idx.space),
            ]
            for idx in indices
        ],
        "free": free,
        "terms": [
            encodeTerm(
                term,
                indexIds,
                freeIds,
            )
            for term in expr
        ],
    }

    if PROFILE:
        print(
            f"Class profile for {name}: Order: {order}; Total time: {perf_counter() - start:.6f} s; Terms: {len(expr)}",
            file = sys.stderr,
            flush = True,
        )

    return out

def overlapBlockTerms(name: str) -> dict[str, Any]:
    """
    Encode one overlap block as compact JSON data.

    Notation: C_n -> {left, right, indices, leftFree, rightFree, terms}
    Examples: overlapBlockTerms("C1") emits the compact term table for block C1.
    """
    block = overlapBlock(name)
    expr = tuple(outputExpr(name))
    indices, indexIds = overlapIndexTable(
        block,
        expr,
    )

    left = EXCITATIONS[block.left]
    right = EXCITATIONS[block.right]
    leftFreeIndices = renamedFreeIndices(left, block.unpack[0])
    rightFreeIndices = renamedFreeIndices(right, block.unpack[1])
    leftFree = [indexId(idx, indexIds) for idx in leftFreeIndices]
    rightFree = [indexId(idx, indexIds) for idx in rightFreeIndices]
    freeIds = set(leftFree + rightFree)

    return {
        "left": block.left,
        "right": block.right,
        "indices": [
            [
                idx.name,
                encodeSpace(idx.space),
            ]
            for idx in indices
        ],
        "leftFree": leftFree,
        "rightFree": rightFree,
        "terms": [
            encodeTerm(
                term,
                indexIds,
                freeIds,
            )
            for term in expr
        ],
    }

def residualTermsData(name: str, order: int) -> dict[str, Any]:
    """
    Encode residual term data for one class or all classes.

    Notation:
        --class all -> every ExcitationClass

    Examples:
        residualTermsData("all", 1) emits all first-order residual term data.
        residualTermsData("all", 2) emits all second-order residual term data.
    """
    if order not in (0, 1, 2):
        raise ValueError(f"unsupported residual order {order}")

    if name == "all":
        names = availableExcitations()
    else:
        if name not in EXCITATIONS:
            raise ValueError(f"unknown excitation class {name}")

        names = (name,)

    return {
        "version": 1,
        "order": order,
        "spaceKinds": SPACE_NAMES,
        "tensorKinds": TENSOR_KIND,
        "classes": {
            className: residualClassTerms(
                className,
                order,
            )
            for className in names
        },
    }

def overlapTermsData(name: str) -> dict[str, Any]:
    """
    Encode overlap term data for one block or all blocks.

    Notation: --block all -> every generated overlap block
    Examples: overlapTermsData("all") emits all compact overlap term data.
    """
    if name == "all":
        names = availableBlocks()
    else:
        if name not in availableBlocks():
            raise ValueError(f"unknown overlap block {name}")

        names = (name,)

    return {
        "version": 1,
        "spaceKinds": SPACE_NAMES,
        "tensorKinds": TENSOR_KIND,
        "blocks": {
            block: overlapBlockTerms(block)
            for block in names
        },
    }

def residualTermsJson(
    name: str,
    order: int,
    pretty: bool = False,
) -> str:
    """
    Emit residual term data as JSON.

    Notation:
        residual expression -> compact class-local term IR

    Examples:
        residualTermsJson("all", 0) emits r0terms.json.
        residualTermsJson("all", 1) emits r1terms.json.
        residualTermsJson("all", 2) emits r2terms.json.
    """
    data = residualTermsData(
        name,
        order,
    )

    if pretty:
        return json.dumps(
            data,
            indent = 2,
        ) + "\n"

    return json.dumps(
        data,
        separators = (
            ",",
            ":",
        ),
    ) + "\n"

def overlapTermsJson(
    name: str,
    pretty: bool = False,
) -> str:
    """
    Emit overlap term data as JSON.

    Notation:

    Examples:
    """
    data = overlapTermsData(name)

    if pretty:
        return json.dumps(
            data,
            indent = 2,
        ) + "\n"

    return json.dumps(
        data,
        separators = (
            ",",
            ":",
        ),
    ) + "\n"

def writeResidualTermsJson(
    name: str,
    order: int,
    out,
    pretty: bool = False,
) -> None:
    """
    Write residual term data as JSON.

    For --class all this streams one excitation class at a time so large first-
    and second-order residual files do not need to be materialised as one Python
    dictionary before output.

    Notation:
        R_mu^(t) ---> JSON term table

    Examples:
        writeResidualTermsJson("all", 1, sys.stdout) writes r1terms.json.
        writeResidualTermsJson("all", 2, sys.stdout) writes r2terms.json.
    """
    if name != "all" or pretty:
        data = residualTermsJson(
            name,
            order,
            pretty = pretty,
        )
        if order == 2:
            print(
                f"R2 progress {name}: Phase: json_write_start",
                flush = True,
            )
        out.write(
            data
        )
        if order == 2:
            print(
                f"R2 progress {name}: Phase: json_write_done",
                flush = True,
            )
        return

    out.write("{")
    out.write('"version":1,')
    out.write(f'"order":{order},')
    out.write('"spaceKinds":')
    json.dump(
        SPACE_NAMES,
        out,
        separators = (",", ":"),
    )
    out.write(",")
    out.write('"tensorKinds":')
    json.dump(
        TENSOR_KIND,
        out,
        separators = (",", ":"),
    )
    out.write(",")
    out.write('"classes":{')

    for i, className in enumerate(availableExcitations()):
        print(
            f"generating {className}",
            file = sys.stderr,
            flush = True,
        )

        if i:
            out.write(",")

        json.dump(
            className,
            out,
            separators = (",", ":"),
        )
        out.write(":")

        data = residualClassTerms(
            className,
            order,
        )
        if order == 2:
            print(
                f"R2 progress {className}: Phase: json_write_start",
                flush = True,
            )
        json.dump(
            data,
            out,
            separators = (",", ":"),
        )
        if order == 2:
            print(
                f"R2 progress {className}: Phase: json_write_done",
                flush = True,
            )

        out.flush()

    out.write("}}\n")

def writeOverlapTermsJson(
    name: str,
    out,
    pretty: bool = False,
) -> None:
    """
    Write overlap term data as JSON.

    For --block all this streams one overlap block at a time so the full overlap term file does not
    need to be materialised as one Python dictionary before output.
    """
    if name != "all" or pretty:
        out.write(
            overlapTermsJson(
                name,
                pretty = pretty,
            )
        )
        return

    out.write("{")
    out.write('"version":1,')
    out.write('"spaceKinds":')
    json.dump(
        SPACE_NAMES,
        out,
        separators = (",", ":"),
    )
    out.write(",")
    out.write('"tensorKinds":')
    json.dump(
        TENSOR_KIND,
        out,
        separators = (",", ":"),
    )
    out.write(",")
    out.write('"blocks":{')

    for i, block in enumerate(availableBlocks()):
        print(
            f"generating {block}",
            file = sys.stderr,
            flush = True,
        )

        if i:
            out.write(",")

        json.dump(
            block,
            out,
            separators = (",", ":"),
        )
        out.write(":")

        json.dump(
            overlapBlockTerms(block),
            out,
            separators = (",", ":"),
        )

        out.flush()

    out.write("}}\n")

def main() -> None:
    """
    Run the term JSON emitter.

    Notation:

    Examples:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kind",
        choices = ("residual", "overlap"),
        default = "residual",
    )

    parser.add_argument(
        "--class",
        dest = "name",
        choices = availableExcitations() + ("all",),
        default = "all",
    )

    parser.add_argument(
        "--order",
        type = int,
        choices = (0, 1, 2),
        default = 0,
    )

    parser.add_argument(
        "--pretty",
        action = "store_true",
    )

    parser.add_argument(
        "--block",
        choices = availableBlocks() + ("all",),
        default = "all",
    )

    parser.add_argument(
        "--profile",
        action = "store_true",
    )

    parser.add_argument(
        "--output",
        default = None,
    )

    args = parser.parse_args()

    if args.kind == "residual" and args.order == 2 and args.output is None:
        raise SystemExit("R2 progress prints to stdout; use --output FILE for JSON output")

    global PROFILE
    PROFILE = PROFILE or args.profile
    configureWick(
        profile = PROFILE,
    )

    if args.output is None:
        out = sys.stdout
        closeOut = False
    else:
        out = open(
            args.output,
            "w",
            encoding = "utf-8",
        )
        closeOut = True

    try:
        if args.kind == "overlap":
            writeOverlapTermsJson(
                args.block,
                out,
                pretty = args.pretty,
            )
        else:
            writeResidualTermsJson(
                args.name,
                args.order,
                out,
                pretty = args.pretty,
            )
    finally:
        if closeOut:
            out.close()

if __name__ == "__main__":
    main()
