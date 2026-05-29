# tools/wick/classes.py

from __future__ import annotations

from dataclasses import dataclass

from symbols import Idx, active, core, virtual


@dataclass(frozen = True)
class SingleSpec:
    """A raw spin-free single excitation operator tau^p_q."""

    create: Idx
    annihilate: Idx


@dataclass(frozen = True)
class DoubleSpec:
    """A raw spin-free double excitation operator tau^{pq}_{rs}."""

    create1: Idx
    create2: Idx
    annihilate1: Idx
    annihilate2: Idx


ExcitationSpec = SingleSpec | DoubleSpec


@dataclass(frozen = True)
class OverlapBlock:
    """A named overlap block.

    The left operator is stored as the raw excitation appearing in the metric:

        <Phi | tau_left^dagger tau_right | Phi>

    The dagger is applied in overlap.py.
    """

    name: str
    label: str
    functionName: str
    left: ExcitationSpec
    right: ExcitationSpec


def c1Block() -> OverlapBlock:
    """C1: C -> A / C -> A."""

    u = active("u")
    i = core("i")
    v = active("v")
    j = core("j")

    return OverlapBlock(
        name = "C1",
        label = "C -> A / C -> A",
        functionName = "overlap_c_to_a",
        left = SingleSpec(
            create = u,
            annihilate = i,
        ),
        right = SingleSpec(
            create = v,
            annihilate = j,
        ),
    )


def c2Block() -> OverlapBlock:
    """C2: A -> V / A -> V."""

    a = virtual("a")
    t = active("t")
    b = virtual("b")
    u = active("u")

    return OverlapBlock(
        name = "C2",
        label = "A -> V / A -> V",
        functionName = "overlap_a_to_v",
        left = SingleSpec(
            create = a,
            annihilate = t,
        ),
        right = SingleSpec(
            create = b,
            annihilate = u,
        ),
    )


def c3Block() -> OverlapBlock:
    """C3: A -> A / A -> A."""

    v = active("v")
    u = active("u")
    x = active("x")
    w = active("w")

    return OverlapBlock(
        name = "C3",
        label = "A -> A / A -> A",
        functionName = "overlap_a_to_a",
        left = SingleSpec(
            create = v,
            annihilate = u,
        ),
        right = SingleSpec(
            create = x,
            annihilate = w,
        ),
    )

def c13Block() -> OverlapBlock:
    """C13: A -> V / AA -> AV."""

    a = virtual("a")
    u = active("u")
    x = active("x")
    b = virtual("b")
    v = active("v")
    w = active("w")

    return OverlapBlock(
        name = "C13",
        label = "A -> V / AA -> AV",
        functionName = "overlap_a_to_v_aa_to_av",
        left = SingleSpec(
            create = a,
            annihilate = u,
        ),
        right = DoubleSpec(
            create1 = x,
            create2 = b,
            annihilate1 = v,
            annihilate2 = w,
        ),
    )


def c14Block() -> OverlapBlock:
    """C14: C -> A / CA -> AA."""

    u = active("u")
    i = core("i")
    w = active("w")
    x = active("x")
    j = core("j")
    v = active("v")

    return OverlapBlock(
        name = "C14",
        label = "C -> A / CA -> AA",
        functionName = "overlap_c_to_a_ca_to_aa",
        left = SingleSpec(
            create = u,
            annihilate = i,
        ),
        right = DoubleSpec(
            create1 = w,
            create2 = x,
            annihilate1 = j,
            annihilate2 = v,
        ),
    )


def c15Block() -> OverlapBlock:
    """C15: A -> A / AA -> AA."""

    u = active("u")
    t = active("t")
    y = active("y")
    z = active("z")
    w = active("w")
    x = active("x")

    return OverlapBlock(
        name = "C15",
        label = "A -> A / AA -> AA",
        functionName = "overlap_a_to_a_aa_to_aa",
        left = SingleSpec(
            create = u,
            annihilate = t,
        ),
        right = DoubleSpec(
            create1 = y,
            create2 = z,
            annihilate1 = w,
            annihilate2 = x,
        ),
    )

def c16Block() -> OverlapBlock:
    """C16: CA -> AV / CA -> VA."""

    w = active("w")
    a = virtual("a")
    i = core("i")
    u = active("u")

    b = virtual("b")
    y = active("y")
    j = core("j")
    x = active("x")

    return OverlapBlock(
        name = "C16",
        label = "CA -> AV / CA -> VA",
        functionName = "overlap_ca_to_av_ca_to_va",
        left = DoubleSpec(
            create1 = w,
            create2 = a,
            annihilate1 = i,
            annihilate2 = u,
        ),
        right = DoubleSpec(
            create1 = b,
            create2 = y,
            annihilate1 = j,
            annihilate2 = x,
        ),
    )

def blockByName(name: str) -> OverlapBlock:
    """Return an overlap block specification by label."""

    blocks = {
        "C1": c1Block,
        "C2": c2Block,
        "C3": c3Block,
        "C13": c13Block,
        "C14": c14Block,
        "C15": c15Block,
        "C16": c16Block,
    }

    key = name.upper()

    if key not in blocks:
        raise ValueError(f"Unsupported overlap block {name}")

    return blocks[key]()


def availableBlocks() -> list[str]:
    """Return supported block names."""

    return ["C1", "C2", "C3", "C13", "C14", "C15", "C16"]
