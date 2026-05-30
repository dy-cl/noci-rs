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

    The left operator is stored as the raw excitation appearing in the metric.
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


def c4Block() -> OverlapBlock:
    """C4: CA -> AV / CA -> AV."""
    v = active("v")
    a = virtual("a")
    i = core("i")
    u = active("u")
    x = active("x")
    b = virtual("b")
    j = core("j")
    w = active("w")

    return OverlapBlock(
        name = "C4",
        label = "CA -> AV / CA -> AV",
        functionName = "overlap_ca_to_av_ca_to_av",
        left = DoubleSpec(
            create1 = v,
            create2 = a,
            annihilate1 = i,
            annihilate2 = u,
        ),
        right = DoubleSpec(
            create1 = x,
            create2 = b,
            annihilate1 = j,
            annihilate2 = w,
        ),
    )


def c5Block() -> OverlapBlock:
    """C5: CA -> VA / CA -> VA."""
    a = virtual("a")
    v = active("v")
    i = core("i")
    u = active("u")
    b = virtual("b")
    x = active("x")
    j = core("j")
    w = active("w")

    return OverlapBlock(
        name = "C5",
        label = "CA -> VA / CA -> VA",
        functionName = "overlap_ca_to_va_ca_to_va",
        left = DoubleSpec(
            create1 = a,
            create2 = v,
            annihilate1 = i,
            annihilate2 = u,
        ),
        right = DoubleSpec(
            create1 = b,
            create2 = x,
            annihilate1 = j,
            annihilate2 = w,
        ),
    )

def c6Block() -> OverlapBlock:
    """C6: CA -> VV / CA -> VV."""
    a = virtual("a")
    b = virtual("b")
    i = core("i")
    u = active("u")
    c = virtual("c")
    d = virtual("d")
    j = core("j")
    v = active("v")

    return OverlapBlock(
        name = "C6",
        label = "CA -> VV / CA -> VV",
        functionName = "overlap_ca_to_vv_ca_to_vv",
        left = DoubleSpec(
            create1 = a,
            create2 = b,
            annihilate1 = i,
            annihilate2 = u,
        ),
        right = DoubleSpec(
            create1 = c,
            create2 = d,
            annihilate1 = j,
            annihilate2 = v,
        ),
    )


def c7Block() -> OverlapBlock:
    """C7: CC -> AV / CC -> AV."""
    u = active("u")
    a = virtual("a")
    i = core("i")
    j = core("j")
    v = active("v")
    b = virtual("b")
    k = core("k")
    l = core("l")

    return OverlapBlock(
        name = "C7",
        label = "CC -> AV / CC -> AV",
        functionName = "overlap_cc_to_av_cc_to_av",
        left = DoubleSpec(
            create1 = u,
            create2 = a,
            annihilate1 = i,
            annihilate2 = j,
        ),
        right = DoubleSpec(
            create1 = v,
            create2 = b,
            annihilate1 = k,
            annihilate2 = l,
        ),
    )


def c8Block() -> OverlapBlock:
    """C8: CC -> AA / CC -> AA."""
    u = active("u")
    v = active("v")
    i = core("i")
    j = core("j")
    w = active("w")
    x = active("x")
    k = core("k")
    l = core("l")

    return OverlapBlock(
        name = "C8",
        label = "CC -> AA / CC -> AA",
        functionName = "overlap_cc_to_aa_cc_to_aa",
        left = DoubleSpec(
            create1 = u,
            create2 = v,
            annihilate1 = i,
            annihilate2 = j,
        ),
        right = DoubleSpec(
            create1 = w,
            create2 = x,
            annihilate1 = k,
            annihilate2 = l,
        ),
    )


def c9Block() -> OverlapBlock:
    """C9: CA -> AA / CA -> AA."""
    v = active("v")
    w = active("w")
    i = core("i")
    u = active("u")
    y = active("y")
    z = active("z")
    j = core("j")
    x = active("x")

    return OverlapBlock(
        name = "C9",
        label = "CA -> AA / CA -> AA",
        functionName = "overlap_ca_to_aa_ca_to_aa",
        left = DoubleSpec(
            create1 = v,
            create2 = w,
            annihilate1 = i,
            annihilate2 = u,
        ),
        right = DoubleSpec(
            create1 = y,
            create2 = z,
            annihilate1 = j,
            annihilate2 = x,
        ),
    )


def c10Block() -> OverlapBlock:
    """C10: AA -> AV / AA -> AV."""
    v = active("v")
    a = virtual("a")
    t = active("t")
    u = active("u")
    z = active("z")
    b = virtual("b")
    x = active("x")
    y = active("y")

    return OverlapBlock(
        name = "C10",
        label = "AA -> AV / AA -> AV",
        functionName = "overlap_aa_to_av_aa_to_av",
        left = DoubleSpec(
            create1 = v,
            create2 = a,
            annihilate1 = t,
            annihilate2 = u,
        ),
        right = DoubleSpec(
            create1 = z,
            create2 = b,
            annihilate1 = x,
            annihilate2 = y,
        ),
    )


def c11Block() -> OverlapBlock:
    """C11: AA -> VV / AA -> VV."""
    a = virtual("a")
    b = virtual("b")
    t = active("t")
    u = active("u")
    c = virtual("c")
    d = virtual("d")
    v = active("v")
    w = active("w")

    return OverlapBlock(
        name = "C11",
        label = "AA -> VV / AA -> VV",
        functionName = "overlap_aa_to_vv_aa_to_vv",
        left = DoubleSpec(
            create1 = a,
            create2 = b,
            annihilate1 = t,
            annihilate2 = u,
        ),
        right = DoubleSpec(
            create1 = c,
            create2 = d,
            annihilate1 = v,
            annihilate2 = w,
        ),
    )


def c12Block() -> OverlapBlock:
    """C12: AA -> AA / AA -> AA."""
    q = active("q")
    s = active("s")
    p = active("p")
    r = active("r")
    t = active("t")
    v = active("v")
    u = active("u")
    w = active("w")

    return OverlapBlock(
        name = "C12",
        label = "AA -> AA / AA -> AA",
        functionName = "overlap_aa_to_aa_aa_to_aa",
        left = DoubleSpec(
            create1 = q,
            create2 = s,
            annihilate1 = p,
            annihilate2 = r,
        ),
        right = DoubleSpec(
            create1 = t,
            create2 = v,
            annihilate1 = u,
            annihilate2 = w,
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
        "C4": c4Block,
        "C5": c5Block,
        "C6": c6Block,
        "C7": c7Block,
        "C8": c8Block,
        "C9": c9Block,
        "C10": c10Block,
        "C11": c11Block,
        "C12": c12Block,
        "C13": c13Block,
        "C14": c14Block,
        "C15": c15Block,
        "C16": c16Block,
    }

    key = name.upper()

    if key not in blocks:
        raise ValueError(f"Unknown block {name}")

    return blocks[key]()


def availableBlocks() -> list[str]:
    """Return supported block names."""
    return [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
    ]
